import torch
import einops
# CPU 环境仍可导入和验证纯 PyTorch 实现。
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:
    if exc.name not in ("triton", "triton.language"):
        raise
    triton = None
    tl = None

from torch import autograd

B_q = 64
B_k = 64


class FlashAttentionTorch(autograd.Function):
    @staticmethod
    def forward(cxt, Q, K, V, is_causal=False):
        # Q: (..., N_q, d), K/V: (..., N_k, d)
        N_q, d = Q.shape[-2], Q.shape[-1]
        N_k = K.shape[-2]
        T_q = (N_q + B_q - 1) // B_q     # 向上取整
        T_k = (N_k + B_k - 1) // B_k

        # 半精度输入使用 float32 计算分数及累加，输出恢复输入类型。
        acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        output = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=acc_dtype)

        for i in range(T_q):
            q_start = i * B_q
            q_end = min(q_start + B_q, N_q)
            # 直接切片，最后一个块可以不足 B_q 行。
            Q_i = Q[..., q_start:q_end, :].to(acc_dtype)
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(Q_i.shape[:-1], device=Q.device, dtype=acc_dtype)
            m_i = torch.full_like(l_i, float('-inf'))

            for j in range(T_k):
                m_i_pre = m_i
                l_i_pre = l_i
                O_i_pre = O_i
                k_start = j * B_k
                k_end = min(k_start + B_k, N_k)
                K_j = K[..., k_start:k_end, :].to(acc_dtype)
                V_j = V[..., k_start:k_end, :].to(acc_dtype)
                # Compute tile of pre-softmax attention scores
                S_i = einops.einsum(Q_i, K_j, "... B_q d, ... B_k d -> ... B_q B_k") / d ** 0.5
                if is_causal:
                    rows = torch.arange(q_start, q_end, device=Q.device)
                    cols = torch.arange(k_start, k_end, device=Q.device)
                    S_i = S_i.masked_fill(rows[:, None] < cols[None, :], float("-inf"))
                m_i = torch.max(m_i_pre, S_i.amax(dim=-1))
                P_i = torch.exp(S_i - m_i.unsqueeze(-1))
                alpha = torch.exp(m_i_pre - m_i)
                l_i = alpha * l_i_pre + torch.sum(P_i, dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i_pre + P_i @ V_j

            O_i = O_i / l_i.unsqueeze(-1)
            l_i = m_i + torch.log(l_i)
            output[..., q_start:q_end, :] = O_i.to(Q.dtype)
            L[..., q_start:q_end] = l_i

        cxt.save_for_backward(Q, K, V, output, L)
        cxt.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """重算每块概率，避免保存或重建完整的 N_q × N_k 注意力矩阵。"""
        Q, K, V, output, L = ctx.saved_tensors
        acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        q, k, v = (x.to(acc_dtype) for x in (Q, K, V))
        do = grad_output.to(acc_dtype)
        dq, dk, dv = (torch.zeros_like(x) for x in (q, k, v))
        # softmax 导数中的行归约 D_i = sum_d(dO_id * O_id)。
        delta = (do * output.to(acc_dtype)).sum(dim=-1)
        scale = Q.shape[-1] ** -0.5
        def tile_derivatives(qs, qe, ks, ke):
            qi, kj = q[..., qs:qe, :], k[..., ks:ke, :]
            scores = (qi @ kj.transpose(-1, -2)) * scale
            if ctx.is_causal:
                rows = torch.arange(qs, qe, device=Q.device)
                cols = torch.arange(ks, ke, device=Q.device)
                scores = scores.masked_fill(rows[:, None] < cols[None, :], float("-inf"))
            prob = torch.exp(scores - L[..., qs:qe, None])
            dp = do[..., qs:qe, :] @ v[..., ks:ke, :].transpose(-1, -2)
            ds = prob * (dp - delta[..., qs:qe, None])
            return prob, ds

        # Algorithm 2 第一阶段：每个 K/V 块独立累加 dK、dV，最后写回一次。
        for ks in range(0, K.shape[-2], B_k):
            ke = min(ks + B_k, K.shape[-2])
            dkj = torch.zeros_like(k[..., ks:ke, :])
            dvj = torch.zeros_like(v[..., ks:ke, :])
            for qs in range(0, Q.shape[-2], B_q):
                qe = min(qs + B_q, Q.shape[-2])
                if ctx.is_causal and ks >= qe:
                    continue
                prob, ds = tile_derivatives(qs, qe, ks, ke)
                dkj += (ds.transpose(-1, -2) @ q[..., qs:qe, :]) * scale
                dvj += prob.transpose(-1, -2) @ do[..., qs:qe, :]
            dk[..., ks:ke, :] = dkj
            dv[..., ks:ke, :] = dvj

        # 第二阶段：每个 Q 块独立累加 dQ，不需要跨块原子加法。
        for qs in range(0, Q.shape[-2], B_q):
            qe = min(qs + B_q, Q.shape[-2])
            dqi = torch.zeros_like(q[..., qs:qe, :])
            for ks in range(0, K.shape[-2], B_k):
                if ctx.is_causal and ks >= qe:
                    break
                ke = min(ks + B_k, K.shape[-2])
                _, ds = tile_derivatives(qs, qe, ks, ke)
                dqi += (ds @ k[..., ks:ke, :]) * scale
            dq[..., qs:qe, :] = dqi
        grads = (dq.to(Q.dtype), dk.to(K.dtype), dv.to(V.dtype), None)
        return grads[:len(ctx.needs_input_grad)]


if triton is not None:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        D_TILE_SIZE: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        O_i = tl.zeros((Q_TILE_SIZE, D_TILE_SIZE), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
        # query 返回, 从起始行号 query_tile_index * Q_TILE_SIZE 开始累加一个块的大小
        query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_offsets = tl.arange(0, K_TILE_SIZE)

        # Q 只加载一次；遍历 K/V 块，在线更新 softmax 的最大值、分母和加权和。
        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            S_i = tl.dot(Q_i, tl.trans(K_j), input_precision="ieee") * scale
            key_indices = j * K_TILE_SIZE + key_offsets
            if is_causal:
                S_i = tl.where(query_indices[:, None] >= key_indices[None, :], S_i, float("-inf"))
            # 补齐的 key 不参与 softmax；补齐的特征列由零填充处理。
            S_i = tl.where(key_indices[None, :] < N_KEYS, S_i, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(S_i, axis=1))
            P_i = tl.exp(S_i - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = alpha * l_i + tl.sum(P_i, axis=1)
            O_i = O_i * alpha[:, None]
            O_i = tl.dot(P_i.to(V_j.dtype), V_j, acc=O_i, input_precision="ieee")
            m_i = m_new

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        O_i = O_i / l_i[:, None]
        L_i = m_i + tl.log(l_i)
        tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, L_i, boundary_check=(0,))


    @triton.jit
    def flash_bwd_delta_kernel(Output, DO, Delta, NQ: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BQ: tl.constexpr):
        """Algorithm 2 第 2 行：D = rowsum(O * dO)，以 FP32 归约。"""
        rows = tl.program_id(0) * BQ + tl.arange(0, BQ)
        batch = tl.program_id(1)
        cols = tl.arange(0, BD)
        offsets = (batch * NQ + rows[:, None]) * D + cols[None, :]
        mask = (rows[:, None] < NQ) & (cols[None, :] < D)
        output = tl.load(Output + offsets, mask, 0).to(tl.float32)
        do = tl.load(DO + offsets, mask, 0).to(tl.float32)
        delta = tl.sum(output * do, axis=1)
        tl.store(Delta + batch * NQ + rows, delta, rows < NQ)


    @triton.jit
    def flash_bwd_dkdv_kernel(
        Q, K, V, DO, L, Delta, DK, DV,
        NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
        BD: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, IS_CAUSAL: tl.constexpr,
    ):
        """每个程序负责一个 K/V 块，遍历 Q 块，独立写回 dK 和 dV。"""
        batch = tl.program_id(1)
        keys = tl.program_id(0) * BK + tl.arange(0, BK)
        features = tl.arange(0, BD)
        kv_offsets = (batch * NK + keys[:, None]) * D + features[None, :]
        kv_mask = (keys[:, None] < NK) & (features[None, :] < D)
        kj = tl.load(K + kv_offsets, kv_mask, 0)
        vj = tl.load(V + kv_offsets, kv_mask, 0)
        dk = tl.zeros((BK, BD), tl.float32)
        dv = tl.zeros((BK, BD), tl.float32)
        scale = D ** -0.5
        # 因果模式下，块末行仍早于最小 key 的 Q 块没有贡献。
        start_q = 0
        if IS_CAUSAL:
            start_q = (tl.program_id(0) * BK // BQ) * BQ
        for qs in range(start_q, NQ, BQ):
            queries = qs + tl.arange(0, BQ)
            q_offsets = (batch * NQ + queries[:, None]) * D + features[None, :]
            q_mask = (queries[:, None] < NQ) & (features[None, :] < D)
            qi = tl.load(Q + q_offsets, q_mask, 0)
            doi = tl.load(DO + q_offsets, q_mask, 0)
            li = tl.load(L + batch * NQ + queries, queries < NQ, 0)
            di = tl.load(Delta + batch * NQ + queries, queries < NQ, 0)
            scores = tl.dot(qi, tl.trans(kj), input_precision="ieee") * scale
            valid = (queries[:, None] < NQ) & (keys[None, :] < NK)
            if IS_CAUSAL:
                valid = valid & (queries[:, None] >= keys[None, :])
            prob = tl.exp(tl.where(valid, scores - li[:, None], float("-inf")))
            dp = tl.dot(doi, tl.trans(vj), input_precision="ieee")
            ds = prob * (dp - di[:, None])
            # tl.dot 的输入类型必须一致；累加器始终为 FP32。
            dv = tl.dot(tl.trans(prob.to(doi.dtype)), doi, acc=dv, input_precision="ieee")
            dk = tl.dot(tl.trans(ds.to(qi.dtype)), qi, acc=dk, input_precision="ieee")
        tl.store(DK + kv_offsets, dk * scale, kv_mask)
        tl.store(DV + kv_offsets, dv, kv_mask)


    @triton.jit
    def flash_bwd_dq_kernel(
        Q, K, V, DO, L, Delta, DQ,
        NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
        BD: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, IS_CAUSAL: tl.constexpr,
    ):
        """每个程序负责一个 Q 块，遍历 K/V 块，独立写回 dQ。"""
        batch = tl.program_id(1)
        queries = tl.program_id(0) * BQ + tl.arange(0, BQ)
        features = tl.arange(0, BD)
        q_offsets = (batch * NQ + queries[:, None]) * D + features[None, :]
        q_mask = (queries[:, None] < NQ) & (features[None, :] < D)
        qi = tl.load(Q + q_offsets, q_mask, 0)
        doi = tl.load(DO + q_offsets, q_mask, 0)
        li = tl.load(L + batch * NQ + queries, queries < NQ, 0)
        di = tl.load(Delta + batch * NQ + queries, queries < NQ, 0)
        dq = tl.zeros((BQ, BD), tl.float32)
        scale = D ** -0.5
        end_k = NK
        if IS_CAUSAL:
            end_k = tl.minimum(NK, (tl.program_id(0) + 1) * BQ)
        for ks in range(0, end_k, BK):
            keys = ks + tl.arange(0, BK)
            kv_offsets = (batch * NK + keys[:, None]) * D + features[None, :]
            kv_mask = (keys[:, None] < NK) & (features[None, :] < D)
            kj = tl.load(K + kv_offsets, kv_mask, 0)
            vj = tl.load(V + kv_offsets, kv_mask, 0)
            scores = tl.dot(qi, tl.trans(kj), input_precision="ieee") * scale
            valid = (queries[:, None] < NQ) & (keys[None, :] < NK)
            if IS_CAUSAL:
                valid = valid & (queries[:, None] >= keys[None, :])
            prob = tl.exp(tl.where(valid, scores - li[:, None], float("-inf")))
            dp = tl.dot(doi, tl.trans(vj), input_precision="ieee")
            ds = prob * (dp - di[:, None])
            dq = tl.dot(ds.to(kj.dtype), kj, acc=dq, input_precision="ieee")
        tl.store(DQ + q_offsets, dq * scale, q_mask)



class FlashAttentionTriton(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """分块计算 output = softmax(QK^T / sqrt(d))V，并保存每行的 logsumexp。"""
        if Q.ndim < 2 or K.ndim != Q.ndim or V.shape != K.shape:
            raise ValueError("Expected Q (..., N_q, d) and K/V (..., N_k, d)")
        if Q.shape[:-2] != K.shape[:-2] or Q.shape[-1] != K.shape[-1]:
            raise ValueError("Q, K, and V must have matching batch and feature dimensions")
        if not Q.is_cuda or Q.device != K.device or Q.device != V.device:
            raise ValueError("Q, K, and V must be on the same CUDA device")
        if Q.dtype != K.dtype or Q.dtype != V.dtype or Q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Expected matching float16, bfloat16, or float32 inputs")
        if Q.numel() == 0 or K.numel() == 0:
            raise ValueError("Attention inputs must be nonempty")

        if triton is None:
            raise RuntimeError("Triton is required for CUDA FlashAttention")
        N_q, d = Q.shape[-2:]
        N_k = K.shape[-2]
        # 将 batch/head 等前导维度合并，二维输入则视为单个 batch。
        q = Q.reshape(-1, N_q, d)
        k = K.reshape(-1, N_k, d)
        v = V.reshape(-1, N_k, d)
        o = torch.empty(q.shape, device=Q.device, dtype=Q.dtype)
        lse = torch.empty(q.shape[:-1], device=Q.device, dtype=torch.float32)
        q_tile_size, k_tile_size = 64, 32
        grid = (triton.cdiv(N_q, q_tile_size), q.shape[0])
        with torch.cuda.device(Q.device):
            flash_fwd_kernel[grid](
                q, k, v, o, lse,
                *q.stride(), *k.stride(), *v.stride(), *o.stride(), *lse.stride(),
                N_q, N_k, d ** -0.5,
                D=d,
                D_TILE_SIZE=max(16, triton.next_power_of_2(d)),
                Q_TILE_SIZE=q_tile_size,
                K_TILE_SIZE=k_tile_size,
                is_causal=is_causal,
            )
        output = o.view(Q.shape)
        L = lse.view(Q.shape[:-1])
        ctx.save_for_backward(Q, K, V, output, L)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, output, L = ctx.saved_tensors
        nq, d = Q.shape[-2:]
        nk = K.shape[-2]
        # 内核按连续的 [batch, sequence, feature] 访问，支持非连续输入和上游梯度。
        q = Q.reshape(-1, nq, d).contiguous()
        k = K.reshape(-1, nk, d).contiguous()
        v = V.reshape(-1, nk, d).contiguous()
        do = grad_output.reshape(-1, nq, d).contiguous()
        o = output.contiguous()
        lse = L.contiguous()
        delta = torch.empty_like(L, dtype=torch.float32)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        bq, bk = B_q, B_k
        bd = max(16, triton.next_power_of_2(d))
        with torch.cuda.device(Q.device):
            flash_bwd_delta_kernel[(triton.cdiv(nq, bq), q.shape[0])](
                o, do, delta, nq, d, bd, bq,
            )
            flash_bwd_dkdv_kernel[(triton.cdiv(nk, bk), q.shape[0])](
                q, k, v, do, lse, delta, dk, dv,
                nq, nk, d, bd, bq, bk, ctx.is_causal,
            )
            flash_bwd_dq_kernel[(triton.cdiv(nq, bq), q.shape[0])](
                q, k, v, do, lse, delta, dq,
                nq, nk, d, bd, bq, bk, ctx.is_causal,
            )
        grads = (dq.reshape(Q.shape), dk.reshape(K.shape), dv.reshape(V.shape), None)
        return grads[:len(ctx.needs_input_grad)]
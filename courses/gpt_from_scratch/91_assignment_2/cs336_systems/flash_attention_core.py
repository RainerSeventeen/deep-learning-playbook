import torch
import einops
import triton
import triton.language as tl

from torch import autograd

B_q = 1
B_k = 1

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
        O = torch.empty_like(Q)
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
                m_i = torch.max(m_i_pre, S_i.amax(dim=-1))
                P_i = torch.exp(S_i - m_i.unsqueeze(-1))
                alpha = torch.exp(m_i_pre - m_i)
                l_i = alpha * l_i_pre + torch.sum(P_i, dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i_pre + P_i @ V_j

            O_i = O_i / l_i.unsqueeze(-1)
            l_i = m_i + torch.log(l_i)
            O[..., q_start:q_end, :] = O_i.to(Q.dtype)
            L[..., q_start:q_end] = l_i

        cxt.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward():
        raise NotImplementedError


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
            S_i += tl.where(query_indices[:, None] >= key_indices[None, :], 0.0, -1e6)
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


class FlashAttentionTriton(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """分块计算 O = softmax(QK^T / sqrt(d))V，并保存每行的 logsumexp。"""
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

        N_q, d = Q.shape[-2:]
        N_k = K.shape[-2]
        # 将 batch/head 等前导维度合并，二维输入则视为单个 batch。
        q = Q.reshape(-1, N_q, d)
        k = K.reshape(-1, N_k, d)
        v = V.reshape(-1, N_k, d)
        o = torch.empty(q.shape, device=Q.device, dtype=Q.dtype)
        l = torch.empty(q.shape[:-1], device=Q.device, dtype=torch.float32)
        q_tile_size, k_tile_size = 64, 32
        grid = (triton.cdiv(N_q, q_tile_size), q.shape[0])
        with torch.cuda.device(Q.device):
            flash_fwd_kernel[grid](
                q, k, v, o, l,
                *q.stride(), *k.stride(), *v.stride(), *o.stride(), *l.stride(),
                N_q, N_k, d ** -0.5,
                D=d,
                D_TILE_SIZE=max(16, triton.next_power_of_2(d)),
                Q_TILE_SIZE=q_tile_size,
                K_TILE_SIZE=k_tile_size,
                is_causal=is_causal,
            )
        O = o.view(Q.shape)
        L = l.view(Q.shape[:-1])
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward is not part of this forward-pass exercise")

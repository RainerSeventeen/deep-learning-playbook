# 第二次作业

## Profiling and Benchmarking

### benchmarking_script

```bash
uv run nsys profile \
  -o profiles/small_s512_full \
  --force-overwrite true \
  --trace=cuda,cublas,cudnn,osrt,nvtx \
  --capture-range=nvtx \
  --nvtx-capture=profile \
  --capture-range-end=stop \
  python -m cs336_systems.benchmark
```



1. Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

>```
>small (cuda)
>forward: mean=57.35ms, std=6.52ms (10 steps)
>backward: mean=121.82ms, std=6.24ms (10 steps)
>optimizer_step: mean=29.47ms, std=4.22ms (10 steps)
>
>medium (cuda)
>forward: mean=157.20ms, std=13.26ms (10 steps)
>backward: mean=335.99ms, std=29.09ms (10 steps)
>optimizer_step: mean=81.49ms, std=7.75ms (10 steps)
>```
>
>再大的就没有测了，因为 24GB 显存 OOM 了。backward 显然是最长消耗时间的。

2. One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

>```
>small (cuda)
>forward: mean=88.83ms, std=100.26ms (10 steps)
>backward: mean=131.66ms, std=30.68ms (10 steps)
>optimizer_step: mean=28.27ms, std=4.31ms (10 steps)
>
>medium (cuda)
>forward: mean=149.96ms, std=9.72ms (10 steps)
>backward: mean=314.55ms, std=18.01ms (10 steps)
>optimizer_step: mean=76.95ms, std=6.48ms (10 steps)
>```
>
>不执行 warm up 的时候，在 small forward 时候会显著增加标准差，如果 warm up 2 个 step 则为下段，可以看到标准差已经显著降低了很多
>
>```
>small (cuda)
>forward: mean=47.13ms, std=455.31us (10 steps)
>backward: mean=100.79ms, std=997.92us (10 steps)
>optimizer_step: mean=23.46ms, std=46.50us (10 steps)
>
>medium (cuda)
>forward: mean=143.33ms, std=433.79us (10 steps)
>backward: mean=300.51ms, std=1.59ms (10 steps)
>optimizer_step: mean=73.42ms, std=52.77us (10 steps)
>```
>
>warm up 主要是为了 CUDA 运行时，显存分配与缓存建立，GPU 时钟升频等等。

### Nsight Systems Profiler

Nsight 是 NV 用来详细检测 GPU 运行情况的库

![运行结果](http://oss.rainerseventeen.cn/blog/2026/202609191731162.png)

1. What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

>forward 中大致都是  48 ms（除了 step 2），和之前的同济保持一致

2. What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

> ![](http://oss.rainerseventeen.cn/blog/2026/202609191750343.png)
>
> 选择 forward 区间并添加 filter，`ampere_sgemm_128x64_tn` 内核占用时间最长，这是一个矩阵运算 kernel，`ampere` 是安培架构（因为 GPU 用的是 3090），`sgemm` 是 Single precision (FP32) General Matrix Multiplication，`tn` 表示矩阵转置。
>
> 如果把 forward 和 backward 全算进去，同样这个 kernel 占用时间最长。

3. Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

>`BinaryFunctor ... MulFunctor<float>` 这一串就是非运算 kernel，用来向量访问内存。

4. Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

> 在一个完整的训练步骤中 `ampere_sgemm_128x64_tn` 的占比是 9.1%，在 推理 forward 过程中占比为 32.6%。

5. Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

> softmax 耗时 161 微秒，attention score 计算耗时 175 微秒。
>
> Attention 计算 $QK^T:(N\times d)(d\times N)$ 为 $2N^2d$ FLOPs，对 $N \times N$ softmax大约可以算作为 $N^2$。
>
> 在如此量级差距下，运算时间几乎相近，可能是由于 GPU 对大规模矩阵乘法的优化。

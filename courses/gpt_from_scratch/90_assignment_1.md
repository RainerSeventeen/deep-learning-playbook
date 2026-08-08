# 第一次作业

> 关于详细的代码实现并不在这里展示，进摘选部分问题进行回答。原作业中的大数据集的训练也相应跳过了。

## BPE Tokenizer

### unicode1

1.  What Unicode character does chr(0) return?

> `'\x00'`

2. How does this character’s string representation (`__repr__()`) differ from its printed representation?

>`repr()` 将它显示为 `'\x00'`，而直接打印时它通常不可见。

3. What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

   ```python
   >>> chr(0)
   >>> print(chr(0)) 
   >>> "this is a test" + chr(0) + "string"
   >>> print("this is a test" + chr(0) + "string")
   ```

> 这是 Unicode 码点 U+0000（NUL）。它在 Python 字符串中是一个正常存在的字符，但直接打印时通常不可见；将其视为 C 字符串结尾是 C 语言及相关 API 的约定，并不意味着 Python 字符串会在这里截断。

### unicode2

1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

> 以字符串 `hello! こんにちは!` 为例：
>
> ```python
> >>> text = "hello! こんにちは!"
> >>> print(text.encode("utf-8"))
> b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
> >>> print(text.encode("utf-16"))
> b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
> ```
>
> UTF-8 兼容 ASCII：常见英文文本仍以单字节表示，适合从字节开始学习 BPE；UTF-16 在英文文本中会频繁出现 `\x00`，并且还涉及字节序与 BOM，容易让字节级统计混入编码层面的规律。UTF-32 的固定四字节表示则更浪费空间。UTF-8 同时能无歧义地表示全部 Unicode 字符，且无需依赖字节序。

2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results. 

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

> ```python
> >>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
> Traceback (most recent call last):
>   File "<stdin>", line 1, in <module>
>   File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
>   File "<stdin>", line 2, in <listcomp>
> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
> ```
>
> 这个函数对每个字节逐一进行解码。UTF-8 的一个字符可能由多个字节组成；把多字节序列拆开后，单独的首字节或续字节不是合法的完整 UTF-8 字符，因此中文等多字节字符会报错。

3. Give a two-byte sequence that does not decode to any Unicode character(s).

>`\xe4\xbd` （"你"的前两个字节）

### train_bpe_tinystories

（关于 `train_bpe` 的实现直接参考代码即可）

1. Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

>总耗时 550s，输入文件大小 2124.55 MiB，词表大小 10,000， RSS 内存占用 12GB
>
>按字节长度统计，最长的三个 token 为 `accomplishment`、`disappointment` 和 `responsibility`（对应词表 ID 分别为 7160、9143、9379）。这些都是 TinyStories 语料中较常见且可复用的词，因此合并为较长 token 是合理的。

2. Profile your code. What part of the tokenizer training process takes the most time?

>在统计并合并的过程中是最慢的，也就是 `bpe_merge()` 的过程

## Transformer

> 课程强烈推荐使用 `einops` 来简化矩阵的形状变化等操作，详细介绍可参见 [Einops 官方文档](https://einops.rocks/)。以及 [Einops 笔记](https://note.rainerseventeen.cn/code-algorithm/api/einops/)

<!-- 1. RoPE 的实现
2. 归一化 softmax 为什么要归一化
3. 如何单次计算实现多头注意力， RoPE 是怎么应用到 MHA 的， MHA 的拆分多头是怎么实现的 -->

### transformer_accounting

1. Consider a GPT-2 XL-sized model using our assignment architecture. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model? Suppose we constructed our model using this configuration:
```
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1,600)
```

>RMSNorm 包含一个 $d_{model}$ 维的可训练缩放参数: $1600$
>
>Transformer Block 中，MHA 的 Q、K、V 和输出投影共有 4 个矩阵，参数量为 $4\times1600\times1600$；SwiGLU 中 3 个权重矩阵的参数量为 $3 \times 1600 \times 4288$；两个 RMSNorm 的缩放参数量为 $2\times1600$。每个 Block 共计 $30,825,600$ 个参数，48 个 Block 共计 $48 \times 30,825,600 = 1,479,628,800$。
>
>Embedding 矩阵参数: $50257 \times 1600 = 80,411,200$
>
>最后 logit 线性层: $1600 \times 50257 = 80,411,200$
>
>全部加起来是 $1,479,628,800 + 80,411,200 + 1,600 + 80,411,200 = 1,640,452,800$，即约 $1.640B$ 参数。
>
>对于 FP32，每个参数占据 4 Bytes；仅加载权重需要 $1,640,452,800 \times 4 = 6,561,811,200$ Bytes，即约 $6.56$ GB（$6.11$ GiB）显存。

2. Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

>一次乘加（multiply-add）计为 2 FLOPs，一次 $A\in R^{a\times b},\quad B\in R^{b\times c}$ 的矩阵乘法算作为 ${2abc\text{ FLOPs}}$。设序列长度 $T=1024$，隐藏维度 $d=1600$，SwiGLU 中间维度 $d_{ff}=4288$。
>
>Embedding 是查表，RMSNorm、RoPE、SwiGLU 激活和 softmax 也都不是矩阵乘法，不计入本题要求的统计
>
>每个 Transformer Block 中：
>
>- Q、K、V 和输出投影共 4 次矩阵乘法$(T, d), (d, d)$，计算量为 $4\times2Td^2=8Td^2$
>- 注意力分数 $QK^\top$ 与注意力权重乘以 $V$ 各一次，计算量为 $2\times2T^2d=4T^2d$。
>- SwiGLU 的 3 个线性投影，计算量为 $3\times2Tdd_{ff}=6Tdd_{ff}$。
>
>因此每个 Block 的矩阵乘法计算量为：
>
>$$
>8\times1024\times1600^2
>+4\times1024^2\times1600
>+6\times1024\times1600\times4288
>=69,835,161,600\ \text{FLOPs}.
>$$
>
>48 个 Block 共需 $48\times69,835,161,600=3,352,087,756,800$ FLOPs。
>
>最后的 logits 线性层为一次 $(T,d)\times(d,\text{vocab\_size})$ 矩阵乘法，计算量为：
>
>$$
>2\times1024\times1600\times50257
>=164,682,137,600\ \text{FLOPs}.
>$$
>
>总计为：
>
>$$
>3,352,087,756,800+164,682,137,600
>=3,516,769,894,400\ \text{FLOPs}
>\approx3.52\ \text{TFLOPs}.
>$$
>
>

3. Based on your analysis above, which parts of the model require the most FLOPs?

> 在本题的 GPT-2 XL 配置（$T=1024$）下，SwiGLU 的 3 个线性投影约为每层 $42.15$B FLOPs，占 Block 矩阵乘法 FLOPs 的约 $60\%$，因此 MLP 是主要开销；Q/K/V/输出投影其次。注意力的 $QK^\top$ 与 $\operatorname{softmax}(QK^\top)V$ 合计约 $6.71$B FLOPs/层，但它随 $T^2$ 增长，在长上下文时会成为主导。

4. Repeat your analysis with GPT-2 small (12 layers, 768 `d_model`, 12 heads), GPT-2 medium (24 layers, 1024 `d_model`, 16 heads), and GPT-2 large (36 layers, 1280 `d_model`, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

> 沿用 $d_{ff}$ 为最接近 $8d/3$ 的 64 的倍数，并取 $T=1024$、词表大小为 50,257，可得：
>
>| 模型 | $d_{ff}$ | 总矩阵乘法 FLOPs |
>| --- | ---: | ---: |
>| GPT-2 small | 2048 | $291,648,307,200\approx0.292$ TFLOPs |
>| GPT-2 medium | 2752 | $830,172,299,264\approx0.830$ TFLOPs |
>| GPT-2 large | 3392 | $1,768,530,903,040\approx1.769$ TFLOPs |
>| GPT-2 XL | 4288 | $3,516,769,894,400\approx3.517$ TFLOPs |
>
> head 数量本身不会改变总计算量：拆分或合并 head 后，投影维度总和仍为 $d$。随着层数和 $d$ 增大，Block 内的投影与 MLP 均按更高阶增长；由于 $d_{ff}\propto d$，MLP 的占比会上升。反之，固定 $T$ 时注意力的 $O(T^2d)$ 占比下降；最终 logits 投影的占比也因它不随层数增长而下降。

5. Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

> 当 $T=16,384$ 时，GPT-2 XL 一次前向的矩阵乘法总计算量为
>
>$$
>133,577,729,638,400\ \text{FLOPs}\approx133.58\ \text{TFLOPs},
>$$
>
>是 $T=1024$ 时的约 $38.0$ 倍。此时各部分占比分别约为：Attention 矩阵乘法 $61.7\%$、MLP $24.2\%$、Q/K/V/输出投影 $12.1\%$、最终 logits 投影 $2.0\%$。原因是 Attention 的两次序列间矩阵乘法按 $T^2$ 增长，其余线性投影仅按 $T$ 增长。

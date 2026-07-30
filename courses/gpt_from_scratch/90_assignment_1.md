# 作业 1

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

> 这个字符在 C 语言中表示的时候字符串结尾，是一个实际存在的字节，在该字符串中表示一个实际存在的字节，但是字符串并不会将其打印出来

### unicode2

1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

> 参考一下该句子`hello! こんにちは! ` 的不同编码
>
> ```python
> >>> print(u8_str)
> b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
> >>> u16 = t_str.encode('utf-16')
> >>> print(u16_str)
> b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
> ```
>
> 仔细观察 utf-16 字符中在英文字符中大量出现 `\x00`的标记，这会影响 BPE 的学习，但是 utf-8 就没有，因为其对 ASCII 的兼容性

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
> 这个函数对每个自己逐一进行解码。由于 utf-8 每一个字节不是和字符一一对应的，对于中文这种多字节的编码就会报错

3. Give a two-byte sequence that does not decode to any Unicode character(s).

>`\xe4\xbd` （"你"的前两个字节）

### train_bpe_tinystories

（关于 `train_bpe` 的实现直接参考代码即可）

1. Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

>总耗时 550s，输入文件大小 2124.55 MiB，词表大小 10,000， RSS 内存占用 12GB
>
>统计可以看到（按照字节数）最长有三个，分别是是 `7160  accomplishment, 9143  disappointment, 9379  responsibility`。

2. Profile your code. What part of the tokenizer training process takes the most time?

>在统计并合并的过程中是最慢的，也就是 `bpe_merge()` 的过程

## Transformer

> 课程强烈推荐使用 `einops` 来简化矩阵的形状变化等操作，详细的介绍文档可以参见[Einops 笔记]() 以及 [Einops 官方文档](https://einops.rocks/)






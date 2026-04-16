# 基础语法

## Python 里的函数是不是对象

是。Python 函数是一等公民（first-class object）：

- 可赋值给变量：`f = print`
- 可作为参数传入：`map(func, lst)`
- 可作为返回值：闭包、装饰器
- 可存入列表/字典

函数类型是 `function`，本质上是 `object` 的实例。

---

## Python 的装饰器是什么，有哪些常见的装饰器？

装饰器是一个**高阶函数**，接收函数作为参数，返回增强后的新函数，用 `@` 语法糖包裹，不修改原函数代码。

```python
def log(func):
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log
def add(a, b):
    return a + b
```

常见装饰器：

| 装饰器 | 作用 |
|--------|------|
| `@staticmethod` | 静态方法，不接收 self/cls |
| `@classmethod` | 类方法，接收 cls |
| `@property` | 将方法转为属性访问 |
| `@functools.wraps` | 保留原函数的 `__name__`、`__doc__` |
| `@functools.lru_cache` | 缓存函数结果（记忆化） |
| `@dataclass` | 自动生成 `__init__`、`__repr__`、`__eq__` |

---

## 全局锁（GIL）是什么，Python 中多线程有什么问题？

**GIL（Global Interpreter Lock）** 是 CPython 解释器的一把全局互斥锁，保证同一时刻只有一个线程执行 Python 字节码。

**多线程的问题：**

- **CPU 密集型任务**：多线程无法真正并行，因为 GIL 不释放，反而因线程切换开销比单线程更慢
- **IO 密集型任务**：线程在等待 IO 时会释放 GIL，多线程有效，可提高吞吐

> CPython 3.13 开始实验性支持禁用 GIL（free-threaded mode）

---

## 怎么写并行，IO 怎么写？

**CPU 密集型 → 多进程**

```python
from multiprocessing import Pool

with Pool(4) as p:
    results = p.map(heavy_func, data)
```

每个进程有独立 GIL，可真正利用多核。

**IO 密集型 → 线程池或 asyncio**

```python
# 线程池（适合调用阻塞 IO 库）
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(fetch, url) for url in urls]

# asyncio（适合大量并发 IO）
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.text()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
```

---

## Python 里数据结构有哪些？

| 数据结构 | 特点 |
|----------|------|
| `list` | 有序、可变、允许重复，O(1) 随机访问 |
| `tuple` | 有序、不可变，比 list 占内存更小 |
| `dict` | 键值对，3.7+ 保持插入顺序，O(1) 查找 |
| `set` | 无序、不重复，O(1) 查找/去重 |
| `collections.deque` | 双端队列，两端 O(1)，适合 BFS |
| `heapq` | 最小堆，O(log n) 插入/弹出 |
| `collections.defaultdict` | 带默认值的 dict |
| `collections.Counter` | 计数器，统计频次 |
| `collections.OrderedDict` | 3.7 前保持顺序用，现已基本被 dict 替代 |

---

## Python 里面的进程和线程有什么区别？DataLoader 中的 num_workers 你是否了解？

**进程 vs 线程：**

| | 进程 | 线程 |
|--|------|------|
| 内存空间 | 独立（fork 后各自一份） | 共享同一进程内存 |
| 通信方式 | IPC：Queue、Pipe、共享内存 | 直接共享变量（需加锁） |
| 创建开销 | 大 | 小 |
| GIL 影响 | 无（各自独立解释器） | 有 |
| 适合场景 | CPU 密集 | IO 密集 |

**DataLoader 的 num_workers：**

- `num_workers=0`：主进程同步加载数据，无并行
- `num_workers=N`：启动 N 个子进程（`multiprocessing`）并行预处理数据，通过共享内存将 batch 传回主进程
- 子进程各自独立，互不干扰，能有效避免 GIL 限制
- 设置过大会因进程间通信和内存占用导致反效果，通常设为 **CPU 核数的一半**

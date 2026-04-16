# Python 基础语法

## 1. python里的函数是不是对象

是。Python 里的函数是对象，而且是一等对象（first-class object）。

- 可以赋值给变量：`f = print`
- 可以作为参数传入其他函数
- 可以作为返回值返回，闭包和装饰器都依赖这一点
- 可以放进列表、字典等容器中

本质上函数也是运行时创建出来的对象，所以 Python 才能把“行为”像“数据”一样传来传去。

---

## 2. python的装饰器是什么， 有哪些常见的装饰器？

装饰器本质上是一个高阶函数：接收函数，返回一个增强后的函数，用来在不改原函数代码的前提下追加逻辑。

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print("calling", func.__name__)
        return func(*args, **kwargs)
    return wrapper

@log_call
def add(a, b):
    return a + b
```

常见装饰器：

| 装饰器 | 作用 |
| --- | --- |
| `@property` | 把方法变成属性访问 |
| `@staticmethod` | 定义静态方法，不自动传 `self` / `cls` |
| `@classmethod` | 定义类方法，第一个参数是 `cls` |
| `@functools.wraps` | 保留被装饰函数的元信息 |
| `@functools.lru_cache` | 给纯函数做结果缓存 |
| `@dataclass` | 自动生成 `__init__`、`__repr__` 等样板代码 |

面试里一般还会补一句：装饰器常用于日志、鉴权、缓存、性能统计和重试。

---

## 3. 全局锁是什么, python 中多线程有什么问题?

全局锁通常指 CPython 里的 GIL（Global Interpreter Lock）。它保证同一时刻只有一个线程执行 Python 字节码。

它带来的核心影响：

- CPU 密集型任务里，多线程通常不能真正利用多核
- 线程切换本身有开销，算力任务甚至可能比单线程更慢
- IO 密集型任务里，线程在等待网络、磁盘时会释放 GIL，所以多线程仍然有价值
- 多线程依然要处理共享变量竞争、锁争用和死锁等并发问题

所以常见经验是：

- CPU 密集型优先多进程
- IO 密集型优先线程池或异步 IO

---

## 4. 怎么写并行, io 怎么写？

先分场景。

- CPU 密集型：用多进程，例如 `multiprocessing` 或 `ProcessPoolExecutor`
- 阻塞式 IO：用线程池，例如 `ThreadPoolExecutor`
- 高并发网络 IO：用 `asyncio` 配合异步库，例如 `aiohttp`

示例：

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(fetch, urls))
```

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.text()

async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

判断原则很简单：看瓶颈是在 CPU，还是在等待外部资源。

---

## 5. python 里数据结构有哪些？

常见内置和标准库数据结构有这些：

| 数据结构 | 特点 |
| --- | --- |
| `list` | 有序、可变、可重复，适合顺序存储和随机访问 |
| `tuple` | 有序、不可变，可作为字典键 |
| `dict` | 键值映射，平均 `O(1)` 查找，工程里最常用 |
| `set` | 无重复集合，适合去重和成员判断 |
| `str` | 不可变字符串，很多场景也要当基础数据结构看 |
| `collections.deque` | 双端队列，头尾插入删除都是 `O(1)` |
| `heapq` | 堆结构，适合优先队列、Top-K |
| `collections.Counter` | 计数器，快速统计频次 |
| `collections.defaultdict` | 带默认值工厂的字典 |

面试里一般会顺带比较几个高频点：

- `list` 和 `tuple`：一个可变，一个不可变
- `dict` 和 `set`：底层都基于哈希表
- `list` 适合顺序遍历，`set` / `dict` 适合快速查找

---

## 6. python 里面的进程和线程有什么区别？dataloader 中的 num_workers 你是否了解？

进程和线程的核心区别是资源隔离和调度粒度不同。

| 维度 | 进程 | 线程 |
| --- | --- | --- |
| 地址空间 | 相互独立 | 共享同一进程内存 |
| 创建和切换开销 | 更大 | 更小 |
| 通信方式 | Queue、Pipe、共享内存等 IPC | 可直接读写共享变量，但要加锁 |
| 是否受 GIL 影响 | 多进程之间互不影响 | CPython 下受 GIL 影响明显 |
| 典型场景 | CPU 密集型 | IO 密集型 |

`DataLoader` 里的 `num_workers` 指加载数据时启动多少个 worker 进程。

- `num_workers=0`：主进程自己读数据和做预处理，最稳定，但吞吐最低
- `num_workers>0`：启动多个子进程并行做数据读取、解码、变换，再把 batch 送回主进程
- 这样做的目的主要是把数据准备和模型训练流水化，减少 GPU 等待数据的时间
- 不是越大越好，过大可能导致进程切换、内存占用和磁盘竞争变严重

实际设置通常看 CPU 核数、磁盘速度、样本预处理复杂度以及 batch size，一般需要压测后定。

# LLM推理框架显存管理机制分析

## 概述

本文档对比分析了vLLM和SGLang两个主流LLM推理框架的显存管理机制，包括缓存策略、内存分配、淘汰算法等关键技术。

## vLLM显存管理机制

### 1. 核心技术：PagedAttention

vLLM采用**PagedAttention**技术，这是一种创新的KV缓存管理方法：

- **分页机制**：将KV缓存分成固定大小的页（通常16-128个token）
- **动态分配**：按需分配和释放内存页，支持非连续内存布局
- **内存效率**：通过虚拟内存映射实现高效的内存利用

### 2. 缓存淘汰策略

vLLM使用**LRU（Least Recently Used）**策略：

```python
# 关键代码片段
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    """Free a list of blocks. The blocks should be ordered by their
    eviction priority, where the first block will be evicted first."""
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    self.free_block_queue.append_n([
        block for block in blocks_list
        if block.ref_cnt == 0 and not block.is_null
    ])
```

**淘汰规则**：
- 优先淘汰最近最少使用的块
- 块按相反顺序添加到空闲队列（尾部有更多哈希token的块先被淘汰）
- 引用计数为0的块才会被添加到空闲队列

### 3. 前缀缓存

vLLM支持基于哈希的前缀缓存：

```python
class BlockHashToBlockMap:
    """Cache of blocks that are used for prefix caching.
    Maps block hash to cached blocks."""
```

- **哈希计算**：基于token序列和父块哈希计算块哈希
- **精确匹配**：通过哈希快速定位已缓存的前缀
- **内存隔离**：支持多租户环境下的缓存隔离

## SGLang显存管理机制

### 1. 核心创新：RadixAttention

SGLang最核心的创新是**RadixAttention**机制：

- **基数树结构**：使用Radix Tree组织KV缓存块
- **前缀匹配**：O(k)复杂度的前缀匹配算法
- **内存层次**：GPU/CPU/外部存储三级缓存架构

### 2. 分层内存池设计

SGLang采用三层内存管理架构：

#### 第一层：请求到Token映射池
```python
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device):
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int32, device=device)
        self.free_slots = list(range(size))
```

#### 第二层：Token到KV池分配器
```python
class BaseTokenToKVPoolAllocator:
    def __init__(self, size, page_size, dtype, device, kvcache):
        self.free_pages = torch.zeros(0, dtype=torch.int64, device=device)
        self.release_pages = torch.zeros(0, dtype=torch.int64, device=device)
```

#### 第三层：KV缓存池
```python
class MHATokenToKVPool(KVCache):
    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device):
        self.k_buffer = torch.zeros((size + page_size, head_num, head_dim), dtype=dtype, device=device)
        self.v_buffer = torch.zeros((size + page_size, head_num, head_dim), dtype=dtype, device=device)
```

### 3. 缓存控制器（CacheController）

SGLang的缓存管理核心：

```python
class CacheController:
    def __init__(self, mem_pool_device_allocator, mem_pool_host):
        self.mem_pool_device_allocator = mem_pool_device_allocator
        self.mem_pool_host = mem_pool_host
        self.write_queue = PriorityQueue()  # 备份队列
        self.load_queue = PriorityQueue()   # 加载队列
```

### 4. 智能淘汰策略

支持LRU和LFU策略：

```python
def evict(self, num_tokens: int):
    leaves = self._collect_leaves()
    eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        priority, node = heapq.heappop(eviction_heap)
        if node.lock_ref > 0:
            continue
        self.token_to_kv_pool_allocator.free(node.value)
        num_evicted += len(node.value)
```

## 对比分析

### 架构差异

| 特性 | vLLM | SGLang |
|------|------|--------|
| **核心机制** | PagedAttention | RadixAttention |
| **缓存结构** | 分页管理 | 基数树 |
| **淘汰策略** | LRU | LRU/LFU |
| **外部存储** | 不支持 | HiCache支持 |
| **内存层次** | 主要GPU | GPU/CPU/远程 |

### 性能特点

**vLLM优势**：
- 内存碎片管理优秀
- 动态内存分配高效
- 实现相对简单

**SGLang优势**：
- 前缀缓存匹配速度更快
- 支持多级存储架构
- 更适合长序列处理

### 内存优化策略

**vLLM**：
- 内存分数静态配置：`--mem-fraction-static 0.8`
- 分块预填充：`--chunked-prefill-size 4096`
- 动态批处理优化

**SGLang**：
- 相同的基础优化策略
- 额外支持外部存储集成
- 更智能的缓存预取机制

## 结论

两个框架都实现了高效的显存管理，但采用了不同的技术路径：

- **vLLM** 更注重内存分配的灵活性和碎片管理
- **SGLang** 更注重缓存匹配的速度和多级存储支持

在实际应用中，选择取决于具体的使用场景和硬件环境。

---

*文档创建时间：2025年*
*基于vLLM和SGLang最新代码分析*

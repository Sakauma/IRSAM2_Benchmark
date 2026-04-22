"""分布式辅助函数。

这里故意只封装最小必要接口：
1. 探测 torchrun 注入的环境变量
2. 初始化 / 销毁进程组
3. 进行对象级 gather / broadcast

上层 runner 只和这些轻量封装交互，不直接散落调用 torch.distributed。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedConfig:
    """描述当前进程在分布式执行中的身份。"""
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def detect_distributed() -> DistributedConfig:
    """从 torchrun 约定的环境变量中探测当前是否启用 DDP。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    return DistributedConfig(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def init_distributed_process_group(distributed: DistributedConfig) -> None:
    """按当前配置初始化进程组。

    如果本来就是单卡，或者进程组已经初始化，则直接返回。
    """
    if not distributed.enabled or dist.is_initialized():
        return
    if torch.cuda.is_available():
        # 多卡时先把当前进程绑定到自己的本地 GPU。
        torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(backend=distributed.backend, rank=distributed.rank, world_size=distributed.world_size)


def destroy_distributed_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """在需要阶段同步时调用，单卡时为空操作。"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(value: Any) -> List[Any]:
    """收集所有 rank 的 Python 对象。

    这里不用 tensor gather，是因为评估行、伪标签样本都属于通用 Python 对象。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered: List[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(value: Any, src: int = 0) -> Any:
    """把主进程计算出的 Python 对象广播给所有 rank。"""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]

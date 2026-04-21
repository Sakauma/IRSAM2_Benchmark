from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def detect_distributed() -> DistributedConfig:
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
    if not distributed.enabled or dist.is_initialized():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(backend=distributed.backend, rank=distributed.rank, world_size=distributed.world_size)


def destroy_distributed_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(value: Any) -> List[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered: List[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]

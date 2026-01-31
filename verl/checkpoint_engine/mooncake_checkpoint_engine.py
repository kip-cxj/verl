# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import time
import gc
from collections import defaultdict
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import ray
import torch
from vllm.distributed.utils import StatelessProcessGroup
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.device import get_torch_device

from mooncake.engine import TransferEngine

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@CheckpointEngineRegistry.register("mooncake")
class MooncakeCheckpointEngine(CheckpointEngine):
    """Mooncake checkpoint engine with p2p communication using TransferEngine

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time.
        device (str): The device to use for the checkpoint engine, "cpu" or "cuda".
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers.
        device_name (str): Mooncake device name filter.
    """

    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        device_name: str = "",
        is_trainer: bool = True,
    ):
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.is_trainer = is_trainer
        self.device_name = device_name
        self.rank = int(os.environ["RANK"])
        self.device_count = get_torch_device().device_count()
        self.local_rank = self.rank % self.device_count
        get_torch_device().set_device(self.local_rank)

        self.engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "ascend_direct" if self.device == "npu" else "rdma",
            self.device_name,
        )
        assert ret == 0, f"TransferEngine initialize failed ret={ret}"

        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{hostname}:{rpc_port}"
        self.hostname = hostname

        self.buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)
        assert self.engine.register_memory(self.buf.data_ptr(), self.bucket_size) == 0, "register_memory failed"

        logger.error(f"yxdebug initialize done pid={os.getpid()} rank={self.rank}")

    def prepare(self) -> str:
        """Prepare send and recv buckets"""
        # self.buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)
        # self.engine.register_memory(self.buf.data_ptr(), self.bucket_size)
        # logger.error(f"yxdebug prepare done session_id={self.session_id} ptr={self.buf.data_ptr():#x} size={self.bucket_size}")
        return self.hostname

    def init_process_group(self, world_size: int, master_addr: str):
        logger.error(f"yxdebug master_addr={master_addr} is_trainer={self.is_trainer} world_size={world_size+1} rank={self.rank}")
        self.world_size = world_size
        if self.is_trainer and self.rank == 0:
            logger.error("yxdebug before tcpstore")
            self.store = StatelessProcessGroup.create(
                host=master_addr,
                port=60010,
                rank=0,
                world_size=world_size+1,
            )
            logger.error("yxdebug after tcpstore")
            buffer_info = {
                "session_id": self.session_id,
                "ptr": self.buf.data_ptr(),
                "len": self.bucket_size,
            }
            self.store.broadcast_obj(obj=buffer_info, src=0)
            logger.error(f"yxdebug session_id={self.session_id} ptr={self.buf.data_ptr():#x} len={self.bucket_size}")
        elif not self.is_trainer:
            logger.error(f"yxdebug before tcpstore pid={os.getpid()}")
            self.store = StatelessProcessGroup.create(
                host=master_addr,
                port=60010,
                rank=self.rank+1,
                world_size=world_size+1,
            )
            logger.error(f"yxdebug after tcpstore pid={os.getpid()}")
            self.buffer_info = self.store.broadcast_obj(obj=None, src=0)
            logger.error(f"yxdebug session_id={self.buffer_info['session_id']} ptr={self.buffer_info['ptr']:#x}")
        logger.error("yxdebug init_process_group done")

    def finish(self):
        """Cleanup communication and deregister memory"""
        self.store = None
        get_torch_device().empty_cache()
        gc.collect()
        logger.error("yxdebug finish done")

    async def wait_for_complete(self):
        magic = torch.tensor([0xab, 0xdc, 0xef, 0x88], dtype=torch.uint8, device=self.device)
        target = magic.repeat(self.world_size)
        while True:
            if torch.equal(self.buf[:4 * self.world_size], target):
                break
            await asyncio.sleep(0)

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send weights using Mooncake TransferEngine"""
        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0

        logger.error(f"yxdebug rank={self.rank} before send")
        for name, weight in weights:
            if self.rank != 0:
                continue
            weight = weight.to(self.rollout_dtype)

            if offset + weight.nbytes > self.bucket_size:
                get_torch_device().synchronize
                info = {
                    "bucket_meta": bucket_meta,
                    "len": offset,
                    "is_last": False,
                }
                self.store.broadcast_obj(obj=info, src=0)
                logger.error(f"yxdebug send_weight len={info['len']} is_last={info['is_last']}")
                await self.wait_for_complete()
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            self.buf[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        if self.rank != 0:
            logger.error(f"yxdebug rank={self.rank} send weights done")
            return

        get_torch_device().synchronize()
        info = {
            "bucket_meta": bucket_meta,
            "len": offset,
            "is_last": True,
        }
        self.store.broadcast_obj(obj=info, src=0)
        await self.wait_for_complete()
        logger.info(f"send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive weights using Mooncake TransferEngine"""
        logger.error(f"yxdebug rank={self.rank} before recv")
        start_time = time.time()
        total_bytes = 0
        while True:
            info = self.store.broadcast_obj(obj=None, src=0)
            logger.error(f"yxdebug recv_weight buffer_info={self.buffer_info} len={info['len']} is_last={info['is_last']}")
            ret = self.engine.transfer_sync_read(
                self.buffer_info["session_id"],
                self.buf.data_ptr(),
                self.buffer_info["ptr"],
                info["len"],
            )
            assert ret == 0, f"transfer_sync_read failed {ret}"
            total_bytes += info["len"]
            for name, meta in info["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = self.buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            self.buf[:4] = torch.tensor([0xab, 0xdc, 0xef, 0x88], dtype=torch.uint8, device=self.device)

            offset = self.buffer_info["ptr"] + self.rank * 4
            ret = self.engine.transfer_sync_write(
                self.buffer_info["session_id"],
                self.buf.data_ptr(),
                offset,
                4,
            )
            assert ret == 0, f"transfer_sync_write failed {ret}"
            if info["is_last"]:
                break

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )

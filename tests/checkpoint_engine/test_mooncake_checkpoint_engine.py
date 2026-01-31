import logging
import os
import pytest
import ray

# 确保 MooncakeCheckpointEngine 被注册
from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine
from tests.checkpoint_engine.test_utils import (
    create_rollout_worker_group,
    create_trainer_worker_group,
)
from verl.single_controller.ray.base import (
    RayResourcePool,
    split_resource_pool,
)
from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@pytest.mark.skip(reason="temporary skip since our ci environment is not ready")
@pytest.mark.parametrize("device", ["npu", "cpu"])
@pytest.mark.parametrize("num_trainer, num_rollout", [(2, 6)])
def test_mooncake_checkpoint_engine(
    num_trainer,
    num_rollout,
    device,
    num_nodes=1,
    num_ranks_per_node=8,  # 8 ranks per node
    check_allclose=True,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    """
    End-to-end test for MooncakeCheckpointEngine.

    Trainer(rank=0) -> Rollout(rank=1..N)
    """
    model_path = os.path.expanduser(model_path)

    # -------------------------
    # 1. Init Ray (NO UCX / GPU required)
    # -------------------------
    ray.init(
        runtime_env={
            "env_vars": {
                "VERL_LOGGING_LEVEL": "DEBUG",
                "ASCEND_USE_SHORT_CONNECTION": "1",
            }
        }
    )

    # -------------------------
    # 2. Resource pool
    # -------------------------
    # Mooncake 不依赖 GPU, 直接用 CPU rank 数量
    resource_pool = RayResourcePool(
        process_on_nodes=[num_ranks_per_node] * num_nodes,
        max_colocate_count=3,
    )

    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(
        resource_pool,
        [num_trainer, num_rollout],
    )

    # -------------------------
    # 3. Mooncake checkpoint args
    # -------------------------
    trainer_checkpoint_kwargs = {
        "bucket_size": 2 * 1024 * 1024 * 1024,
        "device": device,
        "is_trainer": True,
    }

    rollout_checkpoint_kwargs = {
        "bucket_size": 2 * 1024 * 1024 * 1024,
        "device": device,
        "is_trainer": False,
    }
    # -------------------------
    # 4. Create workers
    # -------------------------
    trainer = create_trainer_worker_group(
        model_path,
        trainer_pool,
        checkpoint_backend="mooncake",
        checkpoint_kwargs=trainer_checkpoint_kwargs,
    )
    trainer.reset()

    rollout = create_rollout_worker_group(
        model_path,
        rollout_pool,
        checkpoint_backend="mooncake",
        checkpoint_kwargs=rollout_checkpoint_kwargs,
        device=device,
        check_allclose=check_allclose,
    )

    # -------------------------
    # 5. Run multiple rounds
    # -------------------------
    for _ in range(3):
        # (1) prepare
        hostname = ray.get(
            trainer.execute_checkpoint_engine(["prepare"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )
        logger.info(f"yxdebug here3 master_addr:{hostname[0]}")

        trainer_kwargs = {
            "method": ["init_process_group"] * trainer.world_size,
            "world_size": [rollout.world_size] * trainer.world_size,
            "master_addr": [hostname[0]] * trainer.world_size,
        }

        rollout_kwargs = {
            "method": ["init_process_group"] * rollout.world_size,
            "world_size": [rollout.world_size] * rollout.world_size,
            "master_addr": [hostname[0]] * rollout.world_size,
        }

        # (2). init process group between all workers
        ray.get(
            trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )

        # (3) send / receive weights
        ray.get(trainer.update_weights() + rollout.update_weights())

        # (4) finish
        ray.get(
            trainer.execute_checkpoint_engine(["finish"] * trainer.world_size)
            + rollout.execute_checkpoint_engine(["finish"] * rollout.world_size)
        )

        # (5) correctness check
        rollout.check_weights()

    ray.shutdown()


if __name__ == "__main__":
    test_mooncake_checkpoint_engine(
        num_trainer=8,
        num_rollout=8,
        device="npu",
        num_nodes=1,
        num_ranks_per_node=16,
        check_allclose=True,
        model_path="/mnt/share/Qwen3-8B",
    )

    # test_mooncake_checkpoint_engine(
    #     num_trainer=4,
    #     num_rollout=28,
    #     device="npu",
    #     num_nodes=2,
    #     num_ranks_per_node=16,
    #     check_allclose=False,
    #     model_path="/mnt/share/Qwen3-30B-A3B",
    # )

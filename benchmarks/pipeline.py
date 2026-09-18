import argparse
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from jax_async_ckpt.plugin import JaxAsyncCheckpointer


def generate_synthetic_model(param_dim: int = 4096, num_params: int = 4):
    """Generates a synthetic neural network parameter PyTree with a custom count of parameter tensors.

    Args:
        param_dim: Square matrix size dimension for each parameter (float32).
        num_params: Number of parameter matrices (w1, w2, ..., wN) to generate.
    """
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_params)

    return {
        f"w{i + 1}": jax.random.normal(
            keys[i], (param_dim, param_dim), dtype=jnp.float32
        )
        for i in range(num_params)
    }


@jax.jit
def dummy_train_step(pytree, x):
    """Simulates a heavy compute step (matmuls) updating parameters."""
    pytree = jax.tree_util.tree_map(lambda p: p + 0.001 * jnp.dot(p, x), pytree)
    return pytree


def run_synchronous_checkpoint(pytree, base_filepath: str, rank: int):
    """Blocking synchronous checkpointing (Standard Host Copy + Disk Flush)."""
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    for idx, arr in enumerate(leaves):
        filepath = f"{base_filepath}_sync_rank_{rank}_leaf_{idx}.bin"
        arr_np = np.array(arr)
        with open(filepath, "wb") as f:
            f.write(arr_np.tobytes())


def run_pipeline_benchmark(
    param_dim: int = 4096,
    num_params: int = 4,
    num_steps: int = 10,
    ckpt_interval: int = 2,
    max_buffer_bytes: int = 1024 * 1024 * 1024,
    checkpoint_dir: str | None = None,
    warmup_steps: int = 1,
):
    """Runs pipeline benchmark comparing synchronous vs asynchronous checkpointing.

    Args:
        param_dim: Square matrix size per parameter tensor (float32).
        num_params: Number of parameter matrices.
        num_steps: Total training steps to execute per benchmark run.
        ckpt_interval: Frequency of steps at which a checkpoint is saved.
        max_buffer_bytes: Allocation size for the C++ async engine's staging buffer.
        checkpoint_dir: Directory path for checkpoint files. Uses tempdir if None.
        warmup_steps: Number of initial untimed JIT warmup iterations.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    trace_dir = f"./jax-trace-rank-{rank}"

    rank_mb = (num_params * param_dim**2 * 4) / (1024**2)
    if rank == 0:
        print(f"=== Running Scaling Benchmark across {size} MPI Ranks ===")
        print(
            f"Matrix Dimension: {param_dim} x {param_dim} | Tensors per Rank: {num_params}"
        )
        print(
            f"Model Size per Rank: {rank_mb:.2f} MB | Total System Data: {(rank_mb * size) / 1024:.2f} GB"
        )

    pytree = generate_synthetic_model(param_dim=param_dim, num_params=num_params)

    checkpointer = JaxAsyncCheckpointer(max_buffer_bytes=max_buffer_bytes, comm=comm)
    pytree = generate_synthetic_model(param_dim)
    x = jnp.ones((param_dim, param_dim), dtype=jnp.float32)

    jax.clear_caches()
    # Warmup JIT compilation steps
    for _ in range(warmup_steps):
        pytree = dummy_train_step(pytree, x)
    jax.effects_barrier()

    # Handle directory coordination across ranks
    if checkpoint_dir is None:
        tmpdir_ctx = tempfile.TemporaryDirectory() if rank == 0 else None
        target_dir = comm.bcast(tmpdir_ctx.name if rank == 0 else None, root=0)
    else:
        tmpdir_ctx = None
        target_dir = checkpoint_dir
        if rank == 0:
            os.makedirs(target_dir, exist_ok=True)
        comm.Barrier()

    try:
        jax.profiler.start_trace(trace_dir)
        base_path = os.path.join(target_dir, "bench_run")

        # -------------------------------------------------------------
        # Benchmark 1: Synchronous Checkpointing Loop
        # -------------------------------------------------------------
        comm.Barrier()
        sync_step_times = []
        for step in range(num_steps):
            t0 = time.perf_counter()
            pytree = dummy_train_step(pytree, x)

            if step % ckpt_interval == 0:
                run_synchronous_checkpoint(pytree, f"{base_path}_step_{step}", rank)

            jax.block_until_ready(pytree)
            t1 = time.perf_counter()
            sync_step_times.append((t1 - t0) * 1000)

        # -------------------------------------------------------------
        # Benchmark 2: Asynchronous Pipelined Engine
        # -------------------------------------------------------------
        comm.Barrier()
        async_step_times = []
        for step in range(num_steps):
            t0 = time.perf_counter()
            pytree = dummy_train_step(pytree, x)

            if step % ckpt_interval == 0:
                checkpointer.save_pytree_async(pytree, f"{base_path}_step_{step}")

            jax.block_until_ready(pytree)
            t1 = time.perf_counter()
            async_step_times.append((t1 - t0) * 1000)

        # Final drain of background io_uring writes
        checkpointer.wait_all()

    finally:
        if tmpdir_ctx and rank == 0:
            tmpdir_ctx.cleanup()
        jax.profiler.stop_trace()

    # -------------------------------------------------------------
    # Report Metrics on Rank 0
    # -------------------------------------------------------------
    if rank == 0:
        avg_sync = np.mean(sync_step_times)
        avg_async = np.mean(async_step_times)
        ckpt_sync_avg = np.mean(
            [sync_step_times[i] for i in range(0, num_steps, ckpt_interval)]
        )
        ckpt_async_avg = np.mean(
            [async_step_times[i] for i in range(0, num_steps, ckpt_interval)]
        )

        print("=== Benchmark Results ===")
        print(f"Avg Step Time (Sync):  {avg_sync:.2f} ms")
        print(f"Avg Step Time (Async): {avg_async:.2f} ms")
        print("-----------------------------------")
        print(f"Checkpoint Step Time (Sync):  {ckpt_sync_avg:.2f} ms")
        print(f"Checkpoint Step Time (Async): {ckpt_async_avg:.2f} ms")
        print(
            f"Compute Stall Reduction:       {((ckpt_sync_avg - ckpt_async_avg) / ckpt_sync_avg) * 100:.2f}%"
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Rank Scaling Study Benchmark Pipeline"
    )
    parser.add_argument(
        "--param-dim",
        type=int,
        default=4096,
        help="Parameter matrix size dimension (default: 4096)",
    )
    parser.add_argument(
        "--num-params",
        type=int,
        default=4,
        help="Number of parameter tensors per rank (default: 4)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of benchmark steps to run (default: 10)",
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=2,
        help="Checkpoint interval frequency in steps (default: 2)",
    )
    parser.add_argument(
        "--max-buffer-mb",
        type=int,
        default=1024,
        help="Host buffer size capacity in MB (default: 1024)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: temporary directory)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1, help="Warmup JIT step count (default: 1)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline_benchmark(
        param_dim=args.param_dim,
        num_params=args.num_params,
        num_steps=args.num_steps,
        ckpt_interval=args.ckpt_interval,
        max_buffer_bytes=args.max_buffer_mb * 1024 * 1024,
        checkpoint_dir=args.checkpoint_dir,
        warmup_steps=args.warmup_steps,
    )

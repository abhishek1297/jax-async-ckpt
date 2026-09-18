import os
import time

import jax
from mpi4py import MPI
from test_utils import build_sample_pytree, verify_file_contents

from jax_async_ckpt.plugin import JaxAsyncCheckpointer


def run_dist_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"=== Distributed Integration Test (Ranks: {size}) ===")

    pytree = build_sample_pytree(rank=rank)
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    rank_bytes = sum(arr.nbytes for arr in leaves)
    total_bytes = comm.reduce(rank_bytes, op=MPI.SUM, root=0)

    checkpointer = JaxAsyncCheckpointer(max_buffer_bytes=32 * 1024 * 1024, comm=comm)

    tmpdir = "/tmp/jax_async_ckpt_dist"
    if rank == 0:
        os.makedirs(tmpdir, exist_ok=True)
    comm.Barrier()

    base_path = os.path.join(tmpdir, "ckpt_dist")

    t0 = time.perf_counter()
    checkpointer.save_pytree_async(pytree, base_path)
    checkpointer.wait_all()
    t1 = time.perf_counter()

    if rank == 0:
        duration_ms = (t1 - t0) * 1000
        throughput_gbps = (total_bytes / (1024**3)) / (t1 - t0)
        print(
            f"[SUCCESS] Multi-Rank Checkpoint Completed in {duration_ms:.2f} ms ({throughput_gbps:.2f} GB/s)"
        )

    for idx, arr in enumerate(leaves):
        filepath = f"{base_path}_rank_{rank}_leaf_{idx}.bin"
        assert os.path.exists(filepath), f"Rank {rank} missing file: {filepath}"
        verify_file_contents(filepath, arr)

    comm.Barrier()
    if rank == 0:
        print("SUCCESS: Distributed test passed.")


if __name__ == "__main__":
    run_dist_test()

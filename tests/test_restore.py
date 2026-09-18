import os
import tempfile

from mpi4py import MPI
from test_utils import assert_pytree_allclose, build_sample_pytree

from jax_async_ckpt.plugin import JaxAsyncCheckpointer


def run_restore_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"=== Round-Trip Restoration Test (Ranks: {size}) ===")

    checkpointer = JaxAsyncCheckpointer(max_buffer_bytes=64 * 1024 * 1024, comm=comm)
    original_pytree = build_sample_pytree(rank=rank)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "ckpt_restore")
        manifest_path = f"{base_path}_manifest.json"

        # 1. Save Pass
        checkpointer.save_pytree_async(original_pytree, base_path)
        checkpointer.wait_all()

        if rank == 0:
            assert os.path.exists(manifest_path), "Manifest file not found!"

        comm.Barrier()

        # 2. Restore Pass
        restored_pytree = checkpointer.restore_pytree_async(manifest_path)
        checkpointer.wait_all()

        # 3. Verification
        assert_pytree_allclose(original_pytree, restored_pytree)

    comm.Barrier()
    if rank == 0:
        print("SUCCESS: Round-trip restore test passed.")


if __name__ == "__main__":
    run_restore_test()

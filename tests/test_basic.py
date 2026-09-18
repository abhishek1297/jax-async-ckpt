import os
import tempfile
import time

import jax
from test_utils import build_sample_pytree, verify_file_contents

from jax_async_ckpt.plugin import JaxAsyncCheckpointer


def run_basic_test():
    print("=== Local Integration Test ===")
    pytree = build_sample_pytree(rank=0)
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    total_bytes = sum(arr.nbytes for arr in leaves)

    checkpointer = JaxAsyncCheckpointer(max_buffer_bytes=64 * 1024 * 1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "ckpt_basic")

        t0 = time.perf_counter()
        checkpointer.save_pytree_async(pytree, base_path)
        checkpointer.wait_all()
        t1 = time.perf_counter()

        duration_ms = (t1 - t0) * 1000
        throughput_gbps = (total_bytes / (1024**3)) / (t1 - t0)
        print(
            f"[SUCCESS] Saved {total_bytes / 1024**2:.2f} MB in {duration_ms:.2f} ms ({throughput_gbps:.2f} GB/s)"
        )

        for idx, arr in enumerate(leaves):
            filepath = f"{base_path}_rank_0_leaf_{idx}.bin"
            assert os.path.exists(filepath), f"Missing file: {filepath}"
            verify_file_contents(filepath, arr)

    print("SUCCESS: Basic test passed.")


if __name__ == "__main__":
    run_basic_test()

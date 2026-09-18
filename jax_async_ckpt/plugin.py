import json
import os

import jax
import jax.numpy as jnp

from jax_async_ckpt._core_cpp import AsyncOffloader

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def verify_jax_cuda():
    """Verifies that JAX is compiled with CUDA support and can detect GPU devices."""
    backend = jax.default_backend()
    devices = jax.devices()
    gpu_devices = jax.devices("gpu")

    if backend != "gpu" or not gpu_devices:
        raise RuntimeError(
            f"CUDA JAX Verification Failed!\n"
            f"Current default backend: '{backend}'\n"
            f"Detected devices: {devices}\n"
            f"Ensure JAX with CUDA support is installed. You can install it via:\n"
            f'  pip install --upgrade "jax[cuda13]"'
        )

    return [d.device_kind for d in gpu_devices]


def get_jax_array_pointer(arr: jax.Array) -> int:
    """Extract raw C memory address from a JAX GPU device buffer."""
    device_data = arr.addressable_shards[0].data
    buf = (
        device_data.addressable_data(0)
        if hasattr(device_data, "addressable_data")
        else device_data
    )

    if hasattr(buf, "unsafe_buffer_pointer"):
        return buf.unsafe_buffer_pointer()
    elif hasattr(buf, "__cuda_array_interface__"):
        return buf.__cuda_array_interface__["data"][0]
    elif hasattr(buf, "device_buffer"):
        return buf.device_buffer.unsafe_buffer_pointer()

    raise TypeError(f"Unsupported JAX buffer type for pointer extraction: {type(buf)}")


class JaxAsyncCheckpointer:
    def __init__(self, max_buffer_bytes: int, comm=None):
        verify_jax_cuda()
        self.max_buffer_bytes = max_buffer_bytes
        self.engine = AsyncOffloader(max_buffer_bytes)
        self.comm = comm or (
            MPI.COMM_WORLD if MPI_AVAILABLE and MPI.Is_initialized() else None
        )
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        self._pending_manifest_info = None

    def save_pytree_async(self, pytree, base_filepath: str):
        """Asynchronously saves PyTree tensors to disk using non-blocking I/O pipeline."""
        # Broadcast base_filepath from Rank 0 so all ranks share the exact same output path
        if self.comm:
            base_filepath = self.comm.bcast(base_filepath, root=0)

        self._ensure_directory(base_filepath)
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        self.engine.swap_buffers()

        manifest_leaves = []
        current_offset = 0

        for idx, arr in enumerate(leaves):
            ptr = get_jax_array_pointer(arr)
            nbytes = arr.nbytes
            filepath = f"{base_filepath}_rank_{self.rank}_leaf_{idx}.bin"

            manifest_leaves.append(self._build_leaf_metadata(idx, arr, filepath))

            if nbytes <= self.max_buffer_bytes:
                current_offset = self._offload_standard_leaf(
                    ptr, nbytes, filepath, current_offset
                )
            else:
                self._offload_chunked_leaf(ptr, nbytes, filepath)
                current_offset = 0

        self._pending_manifest_info = {
            "base_filepath": base_filepath,
            "treedef": str(treedef),
            "raw_treedef": treedef,
            "leaves": manifest_leaves,
        }

    def restore_pytree_async(self, manifest_path: str, target_treedef=None):
        """Asynchronously restores PyTree tensors from disk directly into GPU memory."""
        # Broadcast manifest_path from Rank 0 to ensure all ranks read from the same file
        if self.comm:
            manifest_path = self.comm.bcast(manifest_path, root=0)

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        rank_key = f"rank_{self.rank}"
        if rank_key not in manifest["ranks"]:
            raise KeyError(
                f"Rank {self.rank} not found in checkpoint manifest: {manifest_path}"
            )

        leaf_meta_list = manifest["ranks"][rank_key]
        directory = os.path.dirname(manifest_path)

        restored_leaves = []
        self.engine.swap_buffers()
        current_offset = 0

        for meta in leaf_meta_list:
            filepath = os.path.join(directory, meta["filename"])
            shape = tuple(meta["shape"])
            dtype = jnp.dtype(meta["dtype"])
            nbytes = meta["nbytes"]

            gpu_arr = jnp.empty(shape, dtype=dtype)
            ptr = get_jax_array_pointer(gpu_arr)

            if nbytes <= self.max_buffer_bytes:
                current_offset = self._restore_standard_leaf(
                    ptr, nbytes, filepath, current_offset
                )
            else:
                self._restore_chunked_leaf(ptr, nbytes, filepath)
                current_offset = 0

            restored_leaves.append(gpu_arr)

        # Resolve PyTreeDef: use explicit target, pending in-memory reference, or parse manifest string
        treedef = target_treedef
        if treedef is None and self._pending_manifest_info:
            treedef = self._pending_manifest_info.get("raw_treedef")

        if treedef is None and "treedef" in manifest:
            # Reconstruct PyTreeDef from serialized PyTree structure string
            treedef = jax.tree_util.treedef_tuple(())  # Fallback evaluation handle

        if treedef is None:
            raise RuntimeError(
                "Restoration requires an active in-memory PyTreeDef or target_treedef argument."
            )

        return jax.tree_util.tree_unflatten(treedef, restored_leaves)

    def wait_all(self):
        """Blocks until pending io_uring CQEs drain, writes manifest, and syncs all ranks."""
        # 1. Complete local C++ io_uring CQE draining
        self.engine.flush_nvme_writes()

        # 2. Gather metadata and write manifest file
        self._write_manifest_synchronized()

    # -------------------------------------------------------------------------
    # Private Helpers (Pipeline Steps)
    # -------------------------------------------------------------------------

    def _ensure_directory(self, base_filepath: str):
        """Creates target output directory on Root Rank 0 with barrier sync."""
        directory = os.path.dirname(base_filepath)
        if directory and self.rank == 0:
            os.makedirs(directory, exist_ok=True)
        if self.comm:
            self.comm.Barrier()

    def _transfer_chunk_save(
        self, src_ptr: int, size: int, host_offset: int, filepath: str
    ):
        """Executes a single GPU -> Pinned Host -> NVMe write pipeline step."""
        self.engine.offload_gpu_to_host_async(src_ptr, size, host_offset)
        self.engine.wait_gpu_to_host_completion()
        self.engine.write_host_to_nvme_async(filepath, size, host_offset)

    def _transfer_chunk_restore(
        self, dst_ptr: int, size: int, host_offset: int, filepath: str
    ):
        """Executes a single NVMe -> Pinned Host -> GPU read pipeline step."""
        self.engine.read_nvme_to_host_async(filepath, size, host_offset)
        self.engine.flush_nvme_writes()
        self.engine.load_host_to_gpu_async(dst_ptr, size, host_offset)
        self.engine.wait_gpu_to_host_completion()

    def _offload_standard_leaf(
        self, ptr: int, nbytes: int, filepath: str, current_offset: int
    ) -> int:
        """Handles offloading for tensors that fit within standard buffer capacity."""
        if current_offset + nbytes > self.max_buffer_bytes:
            self.engine.flush_nvme_writes()
            self.engine.swap_buffers()
            current_offset = 0

        self._transfer_chunk_save(ptr, nbytes, current_offset, filepath)
        return current_offset + nbytes

    def _offload_chunked_leaf(self, ptr: int, nbytes: int, filepath: str):
        """Streams oversized tensors exceeding single buffer capacity in chunks."""
        bytes_remaining = nbytes
        src_ptr = ptr

        while bytes_remaining > 0:
            chunk = min(bytes_remaining, self.max_buffer_bytes)
            self._transfer_chunk_save(src_ptr, chunk, 0, filepath)
            self.engine.flush_nvme_writes()
            bytes_remaining -= chunk
            src_ptr += chunk

    def _restore_standard_leaf(
        self, ptr: int, nbytes: int, filepath: str, current_offset: int
    ) -> int:
        """Handles restoration for tensors fitting within standard buffer capacity."""
        if current_offset + nbytes > self.max_buffer_bytes:
            self.engine.flush_nvme_writes()
            self.engine.swap_buffers()
            current_offset = 0

        self._transfer_chunk_restore(ptr, nbytes, current_offset, filepath)
        return current_offset + nbytes

    def _restore_chunked_leaf(self, ptr: int, nbytes: int, filepath: str):
        """Streams oversized tensors from NVMe back into GPU memory in chunks."""
        bytes_remaining = nbytes
        dst_ptr = ptr

        while bytes_remaining > 0:
            chunk = min(bytes_remaining, self.max_buffer_bytes)
            self._transfer_chunk_restore(dst_ptr, chunk, 0, filepath)
            bytes_remaining -= chunk
            dst_ptr += chunk

    def _build_leaf_metadata(self, idx: int, arr: jax.Array, filepath: str) -> dict:
        return {
            "leaf_index": idx,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": arr.nbytes,
            "filename": os.path.basename(filepath),
        }

    def _write_manifest_synchronized(self):
        """Collects rank metadata via gather and enforces a hard MPI barrier post-disk write."""
        if self.comm:
            # Collective sync: ALL ranks must enter gather together
            all_manifests = self.comm.gather(self._pending_manifest_info, root=0)
        else:
            all_manifests = [self._pending_manifest_info]

        # Rank 0 handles disk serialization
        if self.rank == 0 and all_manifests and all_manifests[0]:
            base_path = all_manifests[0]["base_filepath"]
            manifest_path = f"{base_path}_manifest.json"

            manifest_data = {
                "num_ranks": self.size,
                "treedef": all_manifests[0]["treedef"],
                "ranks": {
                    f"rank_{i}": m["leaves"] for i, m in enumerate(all_manifests) if m
                },
            }

            # Write and explicitly flush to OS kernel storage cache
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

        # HARD BARRIER: Non-zero ranks MUST wait here until Rank 0 finishes disk I/O
        if self.comm:
            self.comm.Barrier()

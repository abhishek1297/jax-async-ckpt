import jax
import jax.numpy as jnp
import numpy as np


def build_sample_pytree(rank: int = 0) -> dict:
    """Generates a multi-layer PyTree with varied shapes and dtypes."""
    key = jax.random.PRNGKey(42 + rank)
    keys = jax.random.split(key, 4)
    return {
        "layer1": {
            "w": jax.random.normal(keys[0], (1024, 1024), dtype=jnp.float32),
            "b": jax.random.normal(keys[1], (1024,), dtype=jnp.float32),
        },
        "layer2": {
            "w": jax.random.uniform(keys[2], (1024, 256), dtype=jnp.float32),
            "b": jnp.zeros((256,), dtype=jnp.float32),
        },
        "step": jnp.array([100 + rank], dtype=jnp.int32),
    }


def verify_file_contents(filepath: str, expected_array: jax.Array):
    """Verifies that the disk binary file matches source JAX array bytes."""
    expected_bytes = np.array(expected_array).tobytes()

    with open(filepath, "rb") as f:
        read_bytes = f.read(len(expected_bytes))

    assert read_bytes == expected_bytes, f"Binary mismatch in disk file: {filepath}"


def assert_pytree_allclose(original, restored):
    """Recursively validates structure, shape, dtype, and numerical equality."""
    orig_leaves, orig_treedef = jax.tree_util.tree_flatten(original)
    rest_leaves, rest_treedef = jax.tree_util.tree_flatten(restored)

    assert orig_treedef == rest_treedef, "PyTree structure (treedef) mismatch!"
    assert len(orig_leaves) == len(rest_leaves), "Leaf count mismatch!"

    for idx, (orig, rest) in enumerate(zip(orig_leaves, rest_leaves)):
        assert orig.shape == rest.shape, f"Leaf {idx} shape mismatch"
        assert orig.dtype == rest.dtype, f"Leaf {idx} dtype mismatch"
        np.testing.assert_array_equal(
            np.array(orig), np.array(rest), err_msg=f"Leaf {idx} bitwise mismatch!"
        )

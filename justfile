set dotenv-load := true

default: build

sync:
    uv sync --python $(which python3)

build:
    uv pip install --no-deps --no-build-isolation -e .

test: build
    uv run python3 tests/test_basic.py
    uv run mpirun -n 2 python3 tests/test_distributed.py
    uv run mpirun -n 2 python3 tests/test_restore.py

bench: build
    uv run mpirun -n 2 python3 benchmarks/pipeline.py

format:
    uv run ruff check --fix .
    uv run ruff format .
    clang-format -i include/*.hpp src/*.cpp

clean:
    rm -rf .venv build/ _skbuild/ *.egg-info jax_async_ckpt/*.so uv.lock
    uv cache clean

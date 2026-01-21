"""
Generate HDF5 test files for GPU vs CPU benchmarks.

Uses HDF5 1.8 format (libver='earliest') to ensure compatibility with
the GPU implementation which supports superblock v0.
"""

import numpy as np
import h5py
from pathlib import Path

# Output directory
DATA_DIR = Path(__file__).parent / "data"

# Dataset sizes (medium KB range)
SIZE_1D = 10_000      # 10K elements
SIZE_2D_ROWS = 100    # 100x100 = 10K elements
SIZE_2D_COLS = 100

# Random seed for reproducibility
RANDOM_SEED = 42


def create_benchmark_file(filepath: Path, description: str):
    """Create a benchmark HDF5 file with various datasets."""
    print(f"Creating {filepath.name}: {description}")

    # Use earliest format for superblock v0 compatibility
    with h5py.File(filepath, 'w', libver='earliest') as f:
        # 1D double dataset - sequential values for easy verification
        data_1d_double = np.arange(SIZE_1D, dtype=np.float64)
        f.create_dataset('data_1d_double', data=data_1d_double)

        # 1D int32 dataset
        data_1d_int32 = np.arange(SIZE_1D, dtype=np.int32)
        f.create_dataset('data_1d_int32', data=data_1d_int32)

        # 1D float32 dataset
        data_1d_float32 = np.arange(SIZE_1D, dtype=np.float32)
        f.create_dataset('data_1d_float32', data=data_1d_float32)

        # 2D double dataset - value at [i,j] = i * cols + j
        data_2d_double = np.arange(SIZE_2D_ROWS * SIZE_2D_COLS, dtype=np.float64).reshape(SIZE_2D_ROWS, SIZE_2D_COLS)
        f.create_dataset('data_2d_double', data=data_2d_double)

        # 2D int32 dataset
        data_2d_int32 = np.arange(SIZE_2D_ROWS * SIZE_2D_COLS, dtype=np.int32).reshape(SIZE_2D_ROWS, SIZE_2D_COLS)
        f.create_dataset('data_2d_int32', data=data_2d_int32)

        # Small dataset for quick sanity checks
        small_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        f.create_dataset('small', data=small_data)


def create_random_indices_file(filepath: Path):
    """Create a file with pre-generated random indices for random access benchmarks."""
    print(f"Creating {filepath.name}: random access indices")

    rng = np.random.default_rng(RANDOM_SEED)

    with h5py.File(filepath, 'w', libver='earliest') as f:
        # Generate random indices for various counts
        for count in [100, 500, 1000, 5000]:
            indices = rng.integers(0, SIZE_1D, size=count, dtype=np.int64)
            f.create_dataset(f'indices_{count}', data=indices)

        # 2D random indices (flattened)
        for count in [100, 500, 1000, 5000]:
            indices = rng.integers(0, SIZE_2D_ROWS * SIZE_2D_COLS, size=count, dtype=np.int64)
            f.create_dataset(f'indices_2d_{count}', data=indices)


def verify_file_format(filepath: Path):
    """Verify the file uses the expected HDF5 format."""
    with h5py.File(filepath, 'r') as f:
        # Check we can read the superblock version by examining file
        pass  # h5py doesn't expose superblock version directly

    # Read raw bytes to check superblock version
    with open(filepath, 'rb') as f:
        header = f.read(16)
        signature = header[0:8]
        version = header[8]

        expected_sig = bytes([0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a])
        if signature != expected_sig:
            raise ValueError(f"Invalid HDF5 signature in {filepath}")

        print(f"  Superblock version: {version}")
        if version not in (0, 1):
            print(f"  WARNING: Superblock v{version} may not be compatible with GPU implementation")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating HDF5 benchmark test files")
    print("=" * 60)
    print()

    # Main benchmark file
    bench_file = DATA_DIR / "bench_data.h5"
    create_benchmark_file(bench_file, "main benchmark datasets")
    verify_file_format(bench_file)
    print()

    # Random indices file
    indices_file = DATA_DIR / "bench_indices.h5"
    create_random_indices_file(indices_file)
    verify_file_format(indices_file)
    print()

    print("=" * 60)
    print("Summary of generated files:")
    print("=" * 60)
    for f in sorted(DATA_DIR.glob("*.h5")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

    print()
    print("Done! Files ready for benchmarking.")


if __name__ == "__main__":
    main()

"""
Generate HDF5 test files using h5py for testing GPU HDF5 implementation.
Run with: uv run generate_test_files.py
"""

import h5py
import numpy as np
from pathlib import Path
import sys

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def create_basic_test_file(output_dir: Path):
    """Create a basic HDF5 file with various dataset types."""
    filename = output_dir / "test_basic.h5"
    print(f"Creating {filename}...")

    with h5py.File(filename, 'w') as f:
        # Create root attributes
        f.attrs['title'] = 'Basic Test File'
        f.attrs['version'] = 1.0

        # 1D dataset - contiguous storage
        data_1d = np.arange(100, dtype=np.float64)
        f.create_dataset('data_1d', data=data_1d, dtype='f8')

        # 2D dataset - contiguous storage
        data_2d = np.arange(20 * 30, dtype=np.int32).reshape(20, 30)
        f.create_dataset('data_2d', data=data_2d, dtype='i4')

        # Small dataset for testing
        small_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        f.create_dataset('small', data=small_data, dtype='i4')

        # Dataset with fill value
        fill_data = f.create_dataset('with_fill', shape=(10, 10), dtype='f4', fillvalue=-999.0)
        fill_data[5, 5] = 42.0

    print(f"  [OK] Created {filename}")

def create_group_test_file(output_dir: Path):
    """Create HDF5 file with groups and nested structure."""
    filename = output_dir / "test_groups.h5"
    print(f"Creating {filename}...")

    with h5py.File(filename, 'w') as f:
        # Create nested group structure
        g1 = f.create_group('group1')
        g1.attrs['description'] = 'First group'
        g1.create_dataset('dataset_a', data=np.arange(50, dtype=np.float32))

        g2 = f.create_group('group2')
        g2.attrs['description'] = 'Second group'
        g2.create_dataset('dataset_b', data=np.arange(100, 200, dtype=np.int64))

        # Nested groups
        nested = g1.create_group('nested')
        nested.create_dataset('deep_data', data=np.ones((10, 10), dtype=np.float64))

        # Multiple datasets in same group
        for i in range(5):
            g2.create_dataset(f'array_{i}', data=np.full(20, i, dtype=np.int32))

    print(f"  [OK] Created {filename}")

def create_chunked_test_file(output_dir: Path):
    """Create HDF5 file with chunked datasets."""
    filename = output_dir / "test_chunked.h5"
    print(f"Creating {filename}...")

    with h5py.File(filename, 'w') as f:
        # Chunked 1D dataset
        data_1d = np.arange(1000, dtype=np.float32)
        f.create_dataset('chunked_1d', data=data_1d, chunks=(100,))

        # Chunked 2D dataset
        data_2d = np.arange(100 * 200, dtype=np.float64).reshape(100, 200)
        f.create_dataset('chunked_2d', data=data_2d, chunks=(10, 20))

        # Small chunks
        data_small_chunks = np.arange(50 * 50, dtype=np.int32).reshape(50, 50)
        f.create_dataset('small_chunks', data=data_small_chunks, chunks=(5, 5))

    print(f"  [OK] Created {filename}")

def create_datatype_test_file(output_dir: Path):
    """Create HDF5 file with various datatypes."""
    filename = output_dir / "test_datatypes.h5"
    print(f"Creating {filename}...")

    with h5py.File(filename, 'w') as f:
        # Integer types
        f.create_dataset('int8', data=np.array([1, 2, 3, -4, -5], dtype=np.int8))
        f.create_dataset('uint8', data=np.array([1, 2, 3, 4, 5], dtype=np.uint8))
        f.create_dataset('int16', data=np.array([100, -200, 300], dtype=np.int16))
        f.create_dataset('int32', data=np.array([10000, -20000, 30000], dtype=np.int32))
        f.create_dataset('int64', data=np.array([1000000, -2000000], dtype=np.int64))

        # Float types
        f.create_dataset('float32', data=np.array([1.5, 2.5, 3.5], dtype=np.float32))
        f.create_dataset('float64', data=np.array([1.23456789, -9.87654321], dtype=np.float64))

        # Unsigned integers
        f.create_dataset('uint32', data=np.array([100, 200, 300], dtype=np.uint32))
        f.create_dataset('uint64', data=np.array([1000, 2000, 3000], dtype=np.uint64))

    print(f"  [OK] Created {filename}")

def create_write_test_file(output_dir: Path):
    """Create a simple file for write testing."""
    filename = output_dir / "test_write_target.h5"
    print(f"Creating {filename}...")

    with h5py.File(filename, 'w') as f:
        # Create a group for testing group creation
        g = f.create_group('existing_group')

        # Create a dataset that we'll read and verify
        initial_data = np.zeros(100, dtype=np.float32)
        f.create_dataset('write_test', data=initial_data, dtype='f4')

        # Create a 2D dataset for hyperslab testing
        data_2d = np.zeros((50, 50), dtype=np.int32)
        f.create_dataset('hyperslab_test', data=data_2d, dtype='i4')

    print(f"  [OK] Created {filename}")

def main():
    # Create test data directory
    test_dir = Path(__file__).parent.parent.parent / "data"
    test_dir.mkdir(exist_ok=True)

    print("Generating HDF5 test files...\n")

    create_basic_test_file(test_dir)
    create_group_test_file(test_dir)
    create_chunked_test_file(test_dir)
    create_datatype_test_file(test_dir)
    create_write_test_file(test_dir)

    print("\n[OK] All test files generated successfully!")
    print(f"  Location: {test_dir.absolute()}")

if __name__ == '__main__':
    main()

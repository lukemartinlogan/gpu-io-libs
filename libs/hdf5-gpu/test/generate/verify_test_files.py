"""
Verify HDF5 files created by GPU implementation using h5py.
Run with: uv run verify_test_files.py
"""

import h5py
import numpy as np
from pathlib import Path
import sys

def verify_created_file(data_dir: Path) -> bool:
    """Verify the file created by GPU implementation."""
    filename = data_dir / "test_created.h5"

    if not filename.exists():
        print(f"[ERROR] File not found: {filename}")
        return False

    print(f"\nVerifying {filename}...")

    try:
        with h5py.File(filename, 'r') as f:
            # Check structure
            print("  Checking file structure...")

            if 'test_group' not in f:
                print("    [ERROR] Group 'test_group' not found")
                return False
            print("    [OK] Group 'test_group' exists")

            test_group = f['test_group']

            if 'my_dataset' not in test_group:
                print("    [ERROR] Dataset 'my_dataset' not found in test_group")
                return False
            print("    [OK] Dataset 'my_dataset' exists")

            # Check dataset properties
            ds = test_group['my_dataset']
            print(f"  Dataset shape: {ds.shape}")
            print(f"  Dataset dtype: {ds.dtype}")

            if ds.shape != (10,):
                print(f"    [ERROR] Expected shape (10,), got {ds.shape}")
                return False
            print("    [OK] Dataset shape correct")

            if ds.dtype != np.float32:
                print(f"    [ERROR] Expected dtype float32, got {ds.dtype}")
                return False
            print("    [OK] Dataset dtype correct")

            # Check values
            print("  Checking dataset values...")
            data = ds[:]
            expected_start = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

            if not np.allclose(data[:5], expected_start):
                print(f"    [ERROR] Data mismatch. Expected {expected_start}, got {data[:5]}")
                return False
            print(f"    [OK] Data values correct: {data[:5]}")

            # Check remaining values (should be zeros or fill value)
            if data.shape[0] > 5:
                print(f"    Remaining values: {data[5:]}")

        print(f"  [OK] {filename} verified successfully!")
        return True

    except Exception as e:
        print(f"  [ERROR] Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_write_test_file(data_dir: Path) -> bool:
    """Verify modifications to test_write_target.h5."""
    filename = data_dir / "test_write_target.h5"

    if not filename.exists():
        print(f"[ERROR] File not found: {filename}")
        return False

    print(f"\nVerifying {filename}...")

    try:
        with h5py.File(filename, 'r') as f:
            if 'write_test' not in f:
                print("    [ERROR] Dataset 'write_test' not found")
                return False

            ds = f['write_test']
            data = ds[:]

            print(f"  Checking written values at indices 10-14...")
            expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
            actual = data[10:15]

            if np.allclose(actual, expected):
                print(f"    [OK] Written values correct: {actual}")
                return True
            else:
                print(f"    [ERROR] Value mismatch. Expected {expected}, got {actual}")
                return False

    except Exception as e:
        print(f"  [ERROR] Error reading file: {e}")
        return False

def list_file_contents(filename: Path):
    """List contents of an HDF5 file."""
    print(f"\n{'='*60}")
    print(f"Contents of {filename.name}:")
    print('='*60)

    try:
        with h5py.File(filename, 'r') as f:
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 {name}: {obj.shape} {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/")

            f.visititems(print_structure)

    except Exception as e:
        print(f"Error reading {filename}: {e}")

def main():
    test_dir = Path(__file__).parent.parent.parent / "data"

    if not test_dir.exists():
        print(f"[ERROR] Test data directory not found: {test_dir}")
        sys.exit(1)

    print("="*60)
    print("HDF5 Test File Verification")
    print("="*60)

    # List all generated files
    print("\nGenerated test files:")
    for hdf5_file in sorted(test_dir.glob("*.h5")):
        print(f"  • {hdf5_file.name}")

    # Show structure of a few files
    for filename in ['test_basic.h5', 'test_groups.h5']:
        filepath = test_dir / filename
        if filepath.exists():
            list_file_contents(filepath)

    # Verify GPU-created files
    print("\n" + "="*60)
    print("Verifying GPU-created/modified files")
    print("="*60)

    all_passed = True

    # Verify written data
    if (test_dir / "test_write_target.h5").exists():
        if not verify_write_test_file(test_dir):
            all_passed = False
    else:
        print("\n[WARN]  test_write_target.h5 not found (skipping write verification)")

    # Verify created file
    if (test_dir / "test_created.h5").exists():
        if not verify_created_file(test_dir):
            all_passed = False

        # Show its contents
        list_file_contents(test_dir / "test_created.h5")
    else:
        print("\n[WARN]  test_created.h5 not found (GPU test may not have run yet)")

    print("\n" + "="*60)
    if all_passed:
        print("[OK] All verifications passed!")
        print("="*60)
        sys.exit(0)
    else:
        print("[FAIL] Some verifications failed")
        print("="*60)
        sys.exit(1)

if __name__ == '__main__':
    main()

# HDF5 Test File Generator

This directory contains Python scripts for generating and verifying HDF5 test files used by the GPU I/O library tests.

## Setup

Install dependencies using `uv`:

```bash
cd test/generate
uv sync
```

## Usage

### Generate Test Files

Create all HDF5 test files in the `data/` directory:

```bash
uv run generate_test_files.py
```

This generates:
- `data/test_basic.h5` - Basic datasets with various types
- `data/test_groups.h5` - Group hierarchies and nested structures
- `data/test_chunked.h5` - Chunked storage layout datasets
- `data/test_datatypes.h5` - Various HDF5 datatypes
- `data/test_write_target.h5` - Template for write testing

### Verify Test Files

Verify that GPU-created HDF5 files are valid:

```bash
uv run verify_test_files.py
```

## Test Files

All generated files are placed in `../../data/` relative to this directory.

## Requirements

- Python 3.10+
- h5py 3.10.0+
- numpy 1.26.0+

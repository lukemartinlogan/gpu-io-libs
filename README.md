# IOWarp: GPU I/O Libraries
Extensions to the IOWarp runtime for I/O Libraries on the GPU.

## Features
- Reimplementation of HDF5 reading/writing in C++
- Methods to access raw data offsets for compact, contiguous, and chunked datasets
- Efficient hyperslab operations to read and write data

## Architecture
- \[De]serialization system for reading and writing raw components of an HDF5 file
- Implementation of reading and writing all relevant object header message
- Object creation and mutation system build upon raw header messages

## Future Expansions
- Redesign datatype system using templates and template specialization to specify how to read and write custom datatypes
- Support newer, less used versions of HDF5 file spec v4 components
- Support variable length types, like strings

## Building
This HDF5 reimplementation does not depend on any parts of the existing HDF5 library. Having HDF5 installed is not necessary.

### Requirements
- CMake 3.28+
- C++20 compatible compiler

Clone the repository:
```bash
git clone https://github.com/lukemartinlogan/gpu-io-libs
cd gpu-io-libs
```

Switch to the implementation branch:
```bash
git switch reimplementation
```

Configure the CMake project:
```bash
cd implementation
cmake -B build -S .
```

Build the implementation:
```bash
cmake --build build
```

## Demo
```cpp
#include "src/hdf5/file.h"
#include "src/hdf5/group.h" 
#include "src/hdf5/dataset.h"

int main() {
    // 1. File and Group Operations
    File file("example.h5");
    Group root = file.RootGroup();
    Group data_group = root.CreateGroup("data");
    
    // 2. Dataset Creation: contiguous storage
    Dataset ds_1d = data_group.CreateDataset("array_1d", {1000}, DatatypeMessage::f32_t);
    
    // Chunked storage
    Dataset ds_2d = data_group.CreateDataset(
      "array_2d",
      {100, 200}, // dims
      DatatypeMessage::f32_t,
      {{50, 100}} // chunk dims
    );
    
    // 3. Writing Data
    std::vector<float> data(500, 1.5f);
    
    // Simple write
    ds_1d.Write<float>(data, 0);
    
    // Hyperslab write: start=[10,20], count=[25,50]
    ds_2d.WriteHyperslab<float>(data, /* start: */ {10, 20}, /* count: */ {25, 10});
    
    // Strided write: every 2nd element with 3x4 blocks
    ds_2d.WriteHyperslab<float>(
      data,
      {0, 0}, // start
      {5, 5}, // count
      {6, 8}, // stride
      {3, 4} // block size
    );
    
    // 4. Reading Data
    // Simple read
    auto result = ds_1d.Read<float>(50, 100);
    
    // Hyperslab read
    auto hyperslab = ds_2d.ReadHyperslab<float>({5, 10}, {20, 30});
    
    // Strided read
    auto strided = ds_2d.ReadHyperslab<float>({0, 0}, {10, 15}, {2, 3});
    
    // 5. Get all chunk offsets for dataset
    auto all_offsets = ds_2d.RawOffsets();
    
    // Get chunks accessed by hyperslab
    auto accessed_chunks = ds_2d.GetHyperslabChunkRawOffsets(
      {10, 20}, {30, 40}, {1, 2}, {5, 5}
    );
    
    for (const auto& [coords, offset, size] : accessed_chunks) {
      // Process chunk at file position 'offset' with 'size' bytes
    }
    
    return 0;
}
```

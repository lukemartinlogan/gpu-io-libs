#pragma once

#include "superblock.h"
#include "../serialization/stdio.h"

// FIXME: struct name
struct FileLink {
    StdioReaderWriter io;
    SuperblockV0 superblock;

    __device__ __host__
    FileLink(StdioReaderWriter file_io, const SuperblockV0& superblock)
        : io(std::move(file_io)), superblock(superblock) {}

    __device__ __host__
    offset_t AllocateAtEOF(len_t size_bytes);
};
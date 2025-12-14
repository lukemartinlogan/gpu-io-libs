#pragma once

#include "superblock.h"
#include "../serialization/stdio.h"
#include "../serialization/serialization.h"

// FIXME: struct name
struct FileLink {
    StdioReaderWriter io;
    SuperblockV0 superblock;

    __device__ __host__
    FileLink(StdioReaderWriter file_io, const SuperblockV0& superblock)
        : io(std::move(file_io)), superblock(superblock) {}

    template<serde::Serializer S>
    __device__ __host__
    offset_t AllocateAtEOF(len_t size_bytes, S& serializer) {
        offset_t addr = superblock.eof_addr;
        superblock.eof_addr += size_bytes;

        // Write the new EOF address to the superblock in the file
        // (offset 40 is where eof_addr is stored in the superblock)
        serializer.SetPosition(superblock.base_addr + 40);
        serde::Write(serializer, superblock.eof_addr);

        serializer.SetPosition(addr);

        return addr;
    }
};
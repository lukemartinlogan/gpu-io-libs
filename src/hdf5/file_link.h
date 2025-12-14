#pragma once

#include "superblock.h"
#include "../serialization/gpu_posix.h"
#include "../iowarp/gpu_context.h"

// FIXME: struct name
struct FileLink {
    int fd;
    iowarp::GpuContext* ctx;
    SuperblockV0 superblock;

    __device__ __host__
    FileLink(int file_descriptor, iowarp::GpuContext* context, const SuperblockV0& sb)
        : fd(file_descriptor), ctx(context), superblock(sb) {}

    ~FileLink() = default;

    __device__ __host__
    [[nodiscard]] GpuPosixReaderWriter MakeRW() const {
        return {fd, ctx};
    }

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
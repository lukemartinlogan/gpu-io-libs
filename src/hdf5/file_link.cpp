#include "file_link.h"

__device__ __host__
offset_t FileLink::AllocateAtEOF(len_t size_bytes) {
    offset_t addr = superblock.eof_addr;
    superblock.eof_addr += size_bytes;

    // TODO: for some reason writing the whole superblock is invalid
    // io.SetPosition(superblock.base_addr);
    // io.WriteComplex(superblock);
    io.SetPosition(superblock.base_addr + 40);
    serde::Write(io, superblock.eof_addr);

    io.SetPosition(addr);

    return addr;
}

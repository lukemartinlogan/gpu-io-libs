#include "file_link.h"

offset_t FileLink::AllocateAtEOF(len_t size_bytes) {
    offset_t addr = superblock.eof_addr;
    superblock.eof_addr += size_bytes;

    // TODO: not necessary to write everything
    io.SetPosition(superblock.base_addr);
    io.WriteComplex(superblock);

    io.SetPosition(addr);

    return addr;
}

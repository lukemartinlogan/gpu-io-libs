#pragma once

#include "superblock.h"
#include "../serialization/stdio.h"

// FIXME: struct name
struct FileLink {
    StdioReaderWriter io;
    SuperblockV0 superblock;

    FileLink(StdioReaderWriter file_io, const SuperblockV0& superblock)
        : io(std::move(file_io)), superblock(superblock) {}

    offset_t AllocateAtEOF(len_t size_bytes);
};
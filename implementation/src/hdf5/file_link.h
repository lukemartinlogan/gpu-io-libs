#pragma once
#include <mutex>

#include "superblock.h"
#include "../serialization/stdio.h"

// FIXME: struct name
struct FileLink {
    StdioReaderWriter io;
    SuperblockV0 superblock;


    FileLink(StdioReaderWriter file_io, const SuperblockV0& superblock)
        : io(std::move(file_io)), superblock(superblock) {}

    std::lock_guard<std::mutex> Lock() {
        return std::lock_guard(mtx);
    }
private:
    std::mutex mtx{};
};
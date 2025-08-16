#pragma once
#include <mutex>

#include "dataset.h"
#include "group.h"
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

class File {
public:
    explicit File(const std::filesystem::path& path);

    Dataset GetDataset(std::string_view dataset_name) const {
        return root_group_.GetDataset(dataset_name);
    }

private:
    StdioReaderWriter file_;
    SuperblockV0 superblock_;

    // root group
    Group root_group_;
};

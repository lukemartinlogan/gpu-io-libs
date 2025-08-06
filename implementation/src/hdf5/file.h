#pragma once
#include "local_heap.h"
#include "object_header.h"
#include "superblock.h"
#include "tree.h"
#include "../serialization/stdio.h"

class File {
public:
    explicit File(const std::filesystem::path& path);

private:
    StdioReader read_;
    SuperblockV0 superblock_;

    // root group
    BTreeNode group_table_;
    LocalHeap local_heap_;
};

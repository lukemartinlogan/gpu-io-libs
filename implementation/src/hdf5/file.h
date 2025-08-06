#pragma once
#include "local_heap.h"
#include "object_header.h"
#include "superblock.h"
#include "tree.h"
#include "../serialization/stdio.h"

class File;

class Dataset {
public:
    explicit Dataset(const ObjectHeader& header);

private:
    DataLayoutMessage layout_{};
    DatatypeMessage type_{};
    DataspaceMessage space_{};
};

class File {
public:
    explicit File(const std::filesystem::path& path);

    Dataset GetDataset(std::string_view dataset_name);

private:
    StdioReader read_;
    SuperblockV0 superblock_;

    // root group
    BTreeNode group_table_;
    LocalHeap local_heap_;
};

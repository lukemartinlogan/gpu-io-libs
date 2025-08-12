#pragma once
#include "dataset.h"
#include "group.h"
#include "superblock.h"
#include "../serialization/stdio.h"

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

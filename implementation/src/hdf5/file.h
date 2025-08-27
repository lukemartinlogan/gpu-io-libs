#pragma once

#include "dataset.h"
#include "group.h"
#include "file_link.h"

class File {
public:
    explicit File(const std::filesystem::path& path);

    [[nodiscard]] Dataset OpenDataset(std::string_view dataset_name) const {
        return root_group_.OpenDataset(dataset_name);
    }

private:
    std::shared_ptr<FileLink> file_link_{};

    // root group
    Group root_group_;
};

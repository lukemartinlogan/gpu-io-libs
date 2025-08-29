#pragma once

#include "dataset.h"
#include "group.h"
#include "file_link.h"

class File {
public:
    explicit File(const std::filesystem::path& path);

    [[nodiscard]] Group RootGroup() {
        return root_group_;
    }

private:
    std::shared_ptr<FileLink> file_link_{};

    // root group
    Group root_group_;
};

#pragma once

#include "group.h"
#include "file_link.h"
#include "types.h"

class File {
public:
    static hdf5::expected<File> New(const std::filesystem::path& path);

    [[nodiscard]] Group RootGroup() {
        return *root_group_;
    }

private:
    File(std::shared_ptr<FileLink> file_link, Group root_group)
        : file_link_(std::move(file_link)), root_group_(std::move(root_group)) {}

    std::shared_ptr<FileLink> file_link_{};

    // root group
    cstd::optional<Group> root_group_;
};

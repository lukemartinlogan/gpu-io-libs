#pragma once

#include "group.h"
#include "file_link.h"
#include "types.h"

class File {
public:
    __device__ __host__
    static hdf5::expected<File> New(const std::filesystem::path& path);

    __device__ __host__
    [[nodiscard]] Group RootGroup() {
        return *root_group_;
    }

    __device__ __host__
    [[nodiscard]] SuperblockV0 GetSuperBlock() const {
        return file_link_->superblock;
    }

private:
    __device__ __host__
    File(std::shared_ptr<FileLink> file_link, Group root_group)
        : file_link_(std::move(file_link)), root_group_(std::move(root_group)) {}

    std::shared_ptr<FileLink> file_link_{};

    // root group
    cstd::optional<Group> root_group_;
};

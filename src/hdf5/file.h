#pragma once

#include "group.h"
#include "file_link.h"
#include "types.h"

class File {
public:
    __device__
    static hdf5::expected<File> New(const char* filename, iowarp::GpuContext* ctx);

    __device__
    [[nodiscard]] Group RootGroup() {
        return *root_group_;
    }

    __device__
    [[nodiscard]] SuperblockV0 GetSuperBlock() const {
        return file_link_->superblock;
    }

    __device__
    ~File() {
        if (file_link_) {
            file_link_->~FileLink();
        }
    }

    __device__
    File(File&& other) noexcept
        : file_link_(other.file_link_), root_group_(cstd::move(other.root_group_))
    {
        other.file_link_ = nullptr;
    }

    File& operator=(File&&) = delete;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

private:
    __device__
    File(FileLink* file_link, Group root_group)
        : file_link_(file_link), root_group_(cstd::move(root_group)) {}

    FileLink* file_link_{};
    cstd::optional<Group> root_group_;
};

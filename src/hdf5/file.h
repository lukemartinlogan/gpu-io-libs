#pragma once

#include "group.h"
#include "file_link.h"
#include "types.h"

class File {
public:
    __device__ __host__
    static hdf5::expected<File> New(const char* filename, iowarp::GpuContext* ctx);

    __device__ __host__
    [[nodiscard]] Group RootGroup() {
        return *root_group_;
    }

    __device__ __host__
    [[nodiscard]] SuperblockV0 GetSuperBlock() const {
        return file_link_->superblock;
    }

    __device__ __host__
    ~File() {
        if (file_link_) {
#ifdef __CUDA_ARCH__
            file_link_->~FileLink();
            free(file_link_);
#else
            cudaFreeHost(file_link_);
#endif
        }
    }

    __device__ __host__
    File(File&& other) noexcept
        : file_link_(other.file_link_), root_group_(std::move(other.root_group_))
    {
        other.file_link_ = nullptr;
    }

    File& operator=(File&&) = delete;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

private:
    __device__ __host__
    File(FileLink* file_link, Group root_group)
        : file_link_(file_link), root_group_(std::move(root_group)) {}

    FileLink* file_link_{};
    cstd::optional<Group> root_group_;
};

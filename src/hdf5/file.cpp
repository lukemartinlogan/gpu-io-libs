#include "file.h"
#include "../iowarp/gpu_posix.h"
#include "../serialization/gpu_posix.h"

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
  #define OPEN_FLAGS (_O_RDWR | _O_BINARY)
  #define OPEN_MODE (_S_IREAD | _S_IWRITE)
#else
  #include <fcntl.h>
  #include <unistd.h>
  #define OPEN_FLAGS (O_RDWR)
  #define OPEN_MODE (0644)
#endif

__device__
hdf5::expected<File> File::New(const char* filename, iowarp::GpuContext* ctx) {
    printf("[HDF5 File] File::New() called for: %s\n", filename);

    printf("[HDF5 File] Opening file via gpu_posix::open()\n");
    int fd = iowarp::gpu_posix::open(filename, OPEN_FLAGS, OPEN_MODE, *ctx);
    printf("[HDF5 File] Open returned fd=%d\n", fd);

    if (fd < 0) {
        printf("[HDF5 File] Failed to open file\n");
        return hdf5::error(hdf5::HDF5ErrorCode::FileOpenFailed, "failed to open file");
    }

    printf("[HDF5 File] Creating GpuPosixReaderWriter\n");
    auto io = GpuPosixReaderWriter(fd, ctx);
    printf("[HDF5 File] Reading superblock\n");
    auto superblock_result = serde::Read<SuperblockV0>(io);
    if (!superblock_result) {
        printf("[HDF5 File] Failed to read superblock\n");
        iowarp::gpu_posix::close(fd, *ctx);
        return cstd::unexpected(superblock_result.error());
    }
    SuperblockV0 superblock = *superblock_result;
    printf("[HDF5 File] Superblock read successfully\n");

    if (superblock.base_addr != 0) {
        printf("[HDF5 File] Non-zero base address not supported\n");
        iowarp::gpu_posix::close(fd, *ctx);
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "non-zero base address not implemented");
    }

    auto file_link_ptr = ctx->allocator_->NewObj<FileLink>(fd, ctx, superblock);
    if (file_link_ptr.IsNull()) {
        iowarp::gpu_posix::close(fd, *ctx);
        return hdf5::error(hdf5::HDF5ErrorCode::FileOpenFailed, "failed to allocate FileLink");
    }
    FileLink* file_link = file_link_ptr.ptr_;

    offset_t root_group_header_addr = file_link->superblock.base_addr + file_link->superblock.root_group_symbol_table_entry_addr.object_header_addr;
    printf("[HDF5 File] Root group header addr: %llu\n", root_group_header_addr);

    printf("[HDF5 File] Creating Object\n");
    auto object_result = Object::New(file_link, root_group_header_addr);
    if (!object_result) {
        file_link->~FileLink();
        return cstd::unexpected(object_result.error());
    }
    printf("[HDF5 File] Object created successfully\n");

    printf("[HDF5 File] Creating root Group\n");
    auto root_group = Group::New(*object_result);
    if (!root_group) {
        file_link->~FileLink();
        return cstd::unexpected(root_group.error());
    }
    printf("[HDF5 File] Root Group created successfully\n");

    printf("[HDF5 File] File::New() completed successfully\n");
    return File(file_link, cstd::move(*root_group));
}

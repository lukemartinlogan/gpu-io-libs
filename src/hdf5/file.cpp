#include "file.h"
#include "../iowarp/gpu_posix.h"

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
    auto superblock = *superblock_result;
    printf("[HDF5 File] Superblock read successfully\n");

    if (superblock.base_addr != 0) {
        printf("[HDF5 File] Non-zero base address not supported\n");
        iowarp::gpu_posix::close(fd, *ctx);
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "non-zero base address not implemented");
    }

    printf("[HDF5 File] Allocating FileLink\n");
    FileLink* file_link;
#ifdef __CUDA_ARCH__
    file_link = static_cast<FileLink*>(malloc(sizeof(FileLink)));
    printf("[HDF5 File] FileLink allocated at %p, constructing...\n", file_link);
    new (file_link) FileLink(fd, ctx, superblock);
    printf("[HDF5 File] FileLink constructed\n");
#else
    cudaHostAlloc(reinterpret_cast<void**>(&file_link), sizeof(FileLink), cudaHostAllocMapped);
    new (file_link) FileLink(fd, ctx, superblock);
#endif

    offset_t root_group_header_addr = file_link->superblock.base_addr + file_link->superblock.root_group_symbol_table_entry_addr.object_header_addr;
    printf("[HDF5 File] Root group header addr: %llu\n", root_group_header_addr);

    printf("[HDF5 File] Creating Object\n");
    auto object_result = Object::New(file_link, root_group_header_addr);
    if (!object_result) {
        printf("[HDF5 File] Failed to create Object\n");
#ifdef __CUDA_ARCH__
        file_link->~FileLink();
        free(file_link);
#else
        cudaFreeHost(file_link);
#endif
        return cstd::unexpected(object_result.error());
    }
    printf("[HDF5 File] Object created successfully\n");

    printf("[HDF5 File] Creating root Group\n");
    auto root_group = Group::New(*object_result);
    if (!root_group) {
        printf("[HDF5 File] Failed to create root Group\n");
#ifdef __CUDA_ARCH__
        file_link->~FileLink();
        free(file_link);
#else
        cudaFreeHost(file_link);
#endif
        return cstd::unexpected(root_group.error());
    }
    printf("[HDF5 File] Root Group created successfully\n");

    printf("[HDF5 File] File::New() completed successfully\n");
    return File(file_link, cstd::move(*root_group));
}

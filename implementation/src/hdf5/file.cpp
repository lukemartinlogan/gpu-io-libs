#include "file.h"

hdf5::expected<File> File::New(const std::filesystem::path& path) {
    std::shared_ptr<FileLink> file_link;

    // scope since file_io and superblock are invalid after make_shared
    {
        StdioReaderWriter file_io(path);

        auto superblock_result = file_io.ReadComplex<SuperblockV0>();
        if (!superblock_result) return cstd::unexpected(superblock_result.error());
        auto superblock = *superblock_result;

        if (superblock.base_addr != 0) {
            return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "non-zero base address not implemented");
        }

        file_link = std::make_shared<FileLink>(std::move(file_io), superblock);
    }

    // read the root group
    offset_t root_group_header_addr = file_link->superblock.base_addr + file_link->superblock.root_group_symbol_table_entry_addr.object_header_addr;

    auto root_group = Group::New(Object(file_link, root_group_header_addr));
    if (!root_group) {
        return cstd::unexpected(root_group.error());
    }

    return File(std::move(file_link), std::move(*root_group));
}

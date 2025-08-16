#include "file.h"

File::File(const std::filesystem::path& path) {
    // scope since file_io and superblock are invalid after make_shared
    {
        StdioReaderWriter file_io(path);

        auto superblock = file_io.ReadComplex<SuperblockV0>();

        if (superblock.base_addr != 0) {
            throw std::logic_error("not implemented");
        }

        file_link_ = std::make_shared<FileLink>(std::move(file_io), superblock);
    }

    // read the root group
    offset_t root_group_header_addr = file_link_->superblock.root_group_symbol_table_entry_addr.object_header_addr;
    file_link_->io.SetPosition(file_link_->superblock.base_addr + root_group_header_addr);

    auto root_header = file_link_->io.ReadComplex<ObjectHeader>();

    if (root_header.messages.size() != 1) {
        throw std::runtime_error("Root group must have exactly one message");
    }

    root_group_ = Group(root_header, file_link_->io);
}

#include "file.h"

File::File(const std::filesystem::path& path): read_(path) {
    superblock_ = read_.ReadComplex<SuperblockV0>();

    if (superblock_.base_addr != 0) {
        throw std::logic_error("not implemented");
    }

    // read the root group
    offset_t root_group_header_addr = superblock_.root_group_symbol_table_entry_addr.object_header_addr;
    read_.SetPosition(superblock_.base_addr + root_group_header_addr);

    auto root_header = read_.ReadComplex<ObjectHeader>();

    if (root_header.messages.size() != 1) {
        throw std::runtime_error("Root group must have exactly one message");
    }

    root_group_ = Group(root_header, read_);
}

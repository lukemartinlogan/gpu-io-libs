#include "file.h"

Dataset::Dataset(const ObjectHeader& header, Deserializer& de)
    : read_(de)
{
    bool found_layout = false, found_type = false, found_space = false;

    for (const ObjectHeaderMessage& msg : header.messages) {
        if (const auto* layout = std::get_if<DataLayoutMessage>(&msg.message)) {
            layout_ = *layout;
            found_layout = true;
        }
        else if (const auto* type = std::get_if<DatatypeMessage>(&msg.message)) {
            type_ = *type;
            found_type = true;
        }
        else if (const auto* space = std::get_if<DataspaceMessage>(&msg.message)) {
            space_ = *space;
            found_space = true;
        }
    }

    if (!found_layout || !found_type || !found_space) {
        throw std::runtime_error("Dataset header does not contain all required messages");
    }
}

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

    const auto* sym_tbl = std::get_if<SymbolTableMessage>(&root_header.messages.front().message);

    if (!sym_tbl) {
        throw std::runtime_error("Root group must have a symbol table message");
    }

    read_.SetPosition(superblock_.base_addr + sym_tbl->b_tree_addr);
    group_table_ = read_.ReadComplex<BTreeNode>();

    read_.SetPosition(superblock_.base_addr + sym_tbl->local_heap_addr);
    local_heap_ = read_.ReadComplex<LocalHeap>();
}

Dataset File::GetDataset(std::string_view dataset_name) {
    if (group_table_.level != 0) {
        throw std::logic_error("traversing tree not implemented");
    }

    const auto* entries = std::get_if<BTreeEntries<BTreeGroupNodeKey>>(&group_table_.entries);

    if (!entries) {
        throw std::runtime_error("Group table does not contain group node keys");
    }

    if (entries->child_pointers.size() != 1) {
        throw std::runtime_error("nodes at level zero should only have one child pointer");
    }

    offset_t sym_tbl_node = entries->child_pointers.front();
    read_.SetPosition(superblock_.base_addr + sym_tbl_node);

    auto node = read_.ReadComplex<SymbolTableNode>();

    for (const auto& entry : node.entries) {
        std::string entry_name = local_heap_.ReadString(entry.link_name_offset);

        if (entry_name == dataset_name) {
            read_.SetPosition(superblock_.base_addr + entry.object_header_addr);

            auto header = read_.ReadComplex<ObjectHeader>();

            return Dataset(header, this->read_);
        }
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Dataset \"{}\" not found", dataset_name));
}

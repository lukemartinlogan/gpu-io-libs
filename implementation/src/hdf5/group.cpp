#include "group.h"

#include "symbol_table.h"

Group::Group(const ObjectHeader& header, Deserializer& de): read_(&de) {
    auto symb_tbl_msg = std::ranges::find_if(
        header.messages,
        [](const auto& msg) {
            return std::holds_alternative<SymbolTableMessage>(msg);
        }
    );

    if (symb_tbl_msg == header.messages.end()) {
        throw std::runtime_error("Object is not a group header");
    }

    auto symb_tbl = std::get<SymbolTableMessage>(symb_tbl_msg->message);

    read_->SetPosition(/* superblock_.base_addr + */ symb_tbl.b_tree_addr);
    table_ = read_->ReadComplex<BTreeNode>();

    read_->SetPosition(/* superblock_.base_addr + */ symb_tbl.local_heap_addr);
    local_heap_ = read_->ReadComplex<LocalHeap>();
}

Dataset Group::GetDataset(std::string_view dataset_name) const {
    if (table_.level != 0) {
        throw std::logic_error("traversing tree not implemented");
    }

    const auto* entries = std::get_if<BTreeEntries<BTreeGroupNodeKey>>(&table_.entries);

    if (!entries) {
        throw std::runtime_error("Group table does not contain group node keys");
    }

    if (entries->child_pointers.size() != 1) {
        throw std::runtime_error("nodes at level zero should only have one child pointer");
    }

    offset_t sym_tbl_node = entries->child_pointers.front();
    read_->SetPosition(/* superblock_.base_addr + */ sym_tbl_node);

    auto node = read_->ReadComplex<SymbolTableNode>();

    for (const auto& entry : node.entries) {
        std::string entry_name = local_heap_.ReadString(entry.link_name_offset);

        if (entry_name == dataset_name) {
            read_.SetPosition(/* superblock_.base_addr + */ entry.object_header_addr);

            auto header = read_.ReadComplex<ObjectHeader>();

            return Dataset(header, this->read_);
        }
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Dataset \"{}\" not found", dataset_name));
}
#include "group.h"

#include "symbol_table.h"

Group::Group(const ObjectHeader& header, const std::shared_ptr<FileLink>& file)
    : file_(file)
{
    auto symb_tbl_msg = std::ranges::find_if(
        header.messages,
        [](const auto& msg) {
            return std::holds_alternative<SymbolTableMessage>(msg.message);
        }
    );

    if (symb_tbl_msg == header.messages.end()) {
        throw std::runtime_error("Object is not a group header");
    }

    auto symb_tbl = std::get<SymbolTableMessage>(symb_tbl_msg->message);

    file_->io.SetPosition(file_->superblock.base_addr + symb_tbl.b_tree_addr);
    table_ = file_->io.ReadComplex<BTreeNode>();

    file_->io.SetPosition(file_->superblock.base_addr + symb_tbl.local_heap_addr);
    local_heap_ = file_->io.ReadComplex<LocalHeap>();
}

Dataset Group::GetDataset(std::string_view dataset_name) const {
    if (const auto header = GetEntryWithName(dataset_name)) {
        return Dataset(*header, file_);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Dataset \"{}\" not found", dataset_name));
}

Group Group::OpenGroup(std::string_view group_name) const {
    if (const auto header = GetEntryWithName(group_name)) {
        return Group(*header, file_);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Group \"{}\" not found", group_name));
}

SymbolTableNode Group::GetSymbolTableNode() const {
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
    file_->io.SetPosition(file_->superblock.base_addr + sym_tbl_node);

    return file_->io.ReadComplex<SymbolTableNode>();
}

std::optional<ObjectHeader> Group::GetEntryWithName(std::string_view name) const {
    SymbolTableNode node = GetSymbolTableNode();

    for (const auto& entry : node.entries) {
        std::string entry_name = local_heap_.ReadString(entry.link_name_offset);

        if (entry_name == name) {
            file_->io.SetPosition(file_->superblock.base_addr + entry.object_header_addr);

            return file_->io.ReadComplex<ObjectHeader>();
        }
    }

    return std::nullopt;
}

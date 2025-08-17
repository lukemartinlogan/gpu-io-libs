#include "group.h"

#include "symbol_table.h"

Group::Group(const Object& object)
    : object_(object)
{
    ObjectHeader header = object.GetHeader();

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

    object_.file->io.SetPosition(object_.file->superblock.base_addr + symb_tbl.b_tree_addr);
    table_ = object_.file->io.ReadComplex<BTreeNode>();

    object_.file->io.SetPosition(object_.file->superblock.base_addr + symb_tbl.local_heap_addr);
    local_heap_ = object_.file->io.ReadComplex<LocalHeap>();
}

Dataset Group::GetDataset(std::string_view dataset_name) const {
    if (const auto object = GetEntryWithName(dataset_name)) {
        return Dataset(*object);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Dataset \"{}\" not found", dataset_name));
}

Group Group::OpenGroup(std::string_view group_name) const {
    if (const auto object = GetEntryWithName(group_name)) {
        return Group(*object);
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
    object_.file->io.SetPosition(object_.file->superblock.base_addr + sym_tbl_node);

    return object_.file->io.ReadComplex<SymbolTableNode>();
}

std::optional<Object> Group::GetEntryWithName(std::string_view name) const {
    SymbolTableNode node = GetSymbolTableNode();

    for (const auto& entry : node.entries) {
        std::string entry_name = local_heap_.ReadString(entry.link_name_offset);

        if (entry_name == name) {
            return Object(object_.file, object_.file->superblock.base_addr + entry.object_header_addr);
        }
    }

    return std::nullopt;
}

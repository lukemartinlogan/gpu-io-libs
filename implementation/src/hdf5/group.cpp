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

    object_.file->io.SetPosition(object_.file->superblock.base_addr + symb_tbl.local_heap_addr);
    auto local_heap = object_.file->io.ReadComplex<LocalHeap>();

    table_ = BTree(symb_tbl.b_tree_addr, object_.file, local_heap);
}

Dataset Group::GetDataset(std::string_view dataset_name) const {
    if (const auto object = Get(dataset_name)) {
        return Dataset(*object);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Dataset \"{}\" not found", dataset_name));
}

Group Group::OpenGroup(std::string_view group_name) const {
    if (const auto object = Get(group_name)) {
        return Group(*object);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Group \"{}\" not found", group_name));
}

std::optional<Object> Group::Get(std::string_view name) const {
    std::optional<offset_t> sym_table_node_ptr = table_.Get(name);

    if (!sym_table_node_ptr) {
        return std::nullopt;
    }

    offset_t base_addr = object_.file->superblock.base_addr;

    object_.file->io.SetPosition(base_addr + *sym_table_node_ptr);
    auto symbol_table_node = object_.file->io.ReadComplex<SymbolTableNode>();

    std::optional<offset_t> entry_addr = symbol_table_node.FindEntry(name, GetLocalHeap(), object_.file->io);

    if (!entry_addr) {
        return std::nullopt;
    }

    return Object(object_.file, base_addr + *entry_addr);
}

SymbolTableNode Group::GetSymbolTableNode() const {
    BTreeNode table = table_.ReadRoot().value();

    if (!table.IsLeaf()) {
        throw std::logic_error("traversing tree not implemented");
    }

    const auto* entries = std::get_if<BTreeEntries<BTreeGroupNodeKey>>(&table.entries);

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
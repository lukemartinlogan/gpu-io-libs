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

Dataset Group::OpenDataset(std::string_view dataset_name) const {
    if (const auto object = Get(dataset_name)) {
        return Dataset(*object);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Dataset \"{}\" not found", dataset_name));
}

Dataset Group::CreateDataset(
    std::string_view dataset_name,
    const std::vector<len_t>& dimension_sizes,
    const DatatypeMessage& type,
    std::optional<std::vector<byte_t>> fill_value
) {
    if (Get(dataset_name)) {
        throw std::runtime_error(std::format("Dataset \"{}\" already exists", dataset_name));
    }

    offset_t name_offset = GetLocalHeap().WriteString(dataset_name, *object_.file);

    Object new_ds = Object::AllocateEmptyAtEOF(128 + 24, object_.file);

    // turn the span of lens into a vec of dimension info
    std::vector<DimensionInfo> dims(dimension_sizes.size());

    for (size_t i = 0; i < dimension_sizes.size(); ++i) {
        dims[i] = DimensionInfo {
            .size = dimension_sizes[i],
            .max_size = dimension_sizes[i],
            .permutation_index = static_cast<uint32_t>(i),
        };
    }

    DataspaceMessage dataspace(dims, true, false);

    new_ds.WriteMessage(dataspace);
    new_ds.WriteMessage(type);

    new_ds.WriteMessage(FillValueMessage {
        .space_alloc_time = FillValueMessage::SpaceAllocTime::kEarly,
        .write_time = FillValueMessage::ValWriteTime::kIfExplicit,
        .fill_value = fill_value,
    });

    len_t dataset_bytes = dataspace.MaxElements() * type.Size();

    offset_t data_alloc = object_.file->AllocateAtEOF(dataset_bytes);

    object_.file->io.SetPosition(data_alloc);

    new_ds.WriteMessage(DataLayoutMessage {
        ContiguousStorageProperty { .address = data_alloc, .size = dataset_bytes }
    });

    new_ds.WriteMessage(ObjectModificationTimeMessage {
        .modification_time = std::chrono::system_clock::now(),
    });

    SymbolTableEntry ent {
        .link_name_offset = name_offset,
        .object_header_addr = new_ds.GetAddress(),
        .cache_ty = SymbolTableEntryCacheType::kNoDataCached,
    };

    SymbolTableNode node { .entries = { ent }, };

    DynamicBufferSerializer ser;
    ser.WriteComplex(node);

    std::vector<byte_t> padding( // TODO: optimize inserts?
        (2 * object_.file->superblock.group_leaf_node_k - node.entries.size()) * 40
    );

    ser.WriteBuffer(padding);

    offset_t node_alloc = object_.file->AllocateAtEOF(ser.buf.size());
    object_.file->io.SetPosition(node_alloc);
    object_.file->io.WriteBuffer(ser.buf);

    table_.InsertGroup(name_offset, node_alloc);

    UpdateBTreePointer();

    return Dataset(new_ds);
}

Group Group::OpenGroup(std::string_view group_name) const {
    if (const auto object = Get(group_name)) {
        return Group(*object);
    }

    // TODO: better error handling
    throw std::runtime_error(std::format("Group \"{}\" not found", group_name));
}

Group Group::CreateGroup(std::string_view name) {
    if (Get(name)) {
        throw std::runtime_error(std::format("Dataset \"{}\" already exists", name));
    }

    offset_t name_offset = GetLocalHeap().WriteString(name, *object_.file);

    Object new_group_obj = Object::AllocateEmptyAtEOF(24 + 24, object_.file);

    BTreeNode::KValues k{
        .leaf = object_.file->superblock.group_leaf_node_k,
        .internal = object_.file->superblock.group_internal_node_k
    };

    auto [heap, heap_offset] = LocalHeap::AllocateNew(*object_.file, 64);

    offset_t empty_offset = heap.WriteString("", *object_.file);

    heap.RewriteToFile(object_.file->io);

    BTreeEntries<BTreeGroupNodeKey> entries{};
    entries.keys.push_back({ empty_offset });

    offset_t root = BTreeNode { .level = 0, .entries = entries }.AllocateAndWrite(*object_.file, k);

    new_group_obj.WriteMessage(SymbolTableMessage {
        .b_tree_addr = root,
        .local_heap_addr = heap_offset
    });

    SymbolTableEntry ent {
        .link_name_offset = name_offset,
        .object_header_addr = new_group_obj.GetAddress(),
        .cache_ty = SymbolTableEntryCacheType::kNoDataCached,
    };

    SymbolTableNode node { .entries = { ent }, };

    DynamicBufferSerializer ser;
    ser.WriteComplex(node);

    std::vector<byte_t> padding( // TODO: optimize inserts?
        (2 * object_.file->superblock.group_leaf_node_k - node.entries.size()) * 40
    );

    ser.WriteBuffer(padding);

    offset_t node_alloc = object_.file->AllocateAtEOF(ser.buf.size());
    object_.file->io.SetPosition(node_alloc);
    object_.file->io.WriteBuffer(ser.buf);

    table_.InsertGroup(name_offset, node_alloc);

    UpdateBTreePointer();

    return Group(new_group_obj);
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

void Group::Insert(std::string_view name, offset_t object_header_ptr) {
    offset_t name_offset = GetLocalHeap().WriteString(name, *object_.file);

    table_.InsertGroup(name_offset, object_header_ptr);

    UpdateBTreePointer();
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

void Group::UpdateBTreePointer() {
    SymbolTableMessage sym = object_.DeleteMessage<SymbolTableMessage>().value();
    sym.b_tree_addr = table_.addr_.value();

    object_.WriteMessage(sym);
}

#include "group.h"

#include "symbol_table.h"

hdf5::expected<Group> Group::New(const Object& object) {
    auto header_result = object.GetHeader();
    if (!header_result) return cstd::unexpected(header_result.error());
    ObjectHeader header = *header_result;

    auto symb_tbl_msg = std::ranges::find_if(
        header.messages,
        [](const auto& msg) {
            return cstd::holds_alternative<SymbolTableMessage>(msg.message);
        }
    );

    if (symb_tbl_msg == header.messages.end()) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Object is not a group header");
    }

    auto symb_tbl = cstd::get<SymbolTableMessage>(symb_tbl_msg->message);

    object.file->io.SetPosition(object.file->superblock.base_addr + symb_tbl.local_heap_addr);
    auto local_heap_result = object.file->io.ReadComplex<LocalHeap>();
    if (!local_heap_result) return cstd::unexpected(local_heap_result.error());

    GroupBTree table(symb_tbl.b_tree_addr, object.file, *local_heap_result);

    return Group(object, std::move(table));
}

hdf5::expected<Dataset> Group::OpenDataset(std::string_view dataset_name) const {
    auto object_result = Get(dataset_name);
    if (!object_result) {
        return cstd::unexpected(object_result.error());
    }

    if (const auto& object = *object_result) {
        return Dataset::New(*object);
    }

    return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Dataset not found");
}

hdf5::expected<Dataset> Group::CreateDataset(
    std::string_view dataset_name,
    const hdf5::dim_vector<len_t>& dimension_sizes,
    const DatatypeMessage& type,
    cstd::optional<hdf5::dim_vector<uint32_t>> chunk_dims,
    cstd::optional<std::vector<byte_t>> fill_value
) {
    auto exists_result = Get(dataset_name);
    if (!exists_result) {
        return cstd::unexpected(exists_result.error());
    }
    if (*exists_result) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Dataset already exists");
    }

    auto name_offset_result = GetLocalHeap().WriteString(dataset_name, *object_.file);
    if (!name_offset_result) {
        return cstd::unexpected(name_offset_result.error());
    }
    offset_t name_offset = *name_offset_result;

    Object new_ds = Object::AllocateEmptyAtEOF(128 + 24, object_.file);

    // turn the span of lens into a vec of dimension info
    hdf5::dim_vector<DimensionInfo> dims(dimension_sizes.size());

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

    if (chunk_dims.has_value()) {
        ChunkedStorageProperty chunked_prop {
            .b_tree_addr = ChunkedBTree::CreateNew(object_.file, dimension_sizes),
            .dimension_sizes = *chunk_dims,
            .elem_size_bytes = static_cast<uint32_t>(type.Size()),
        };

        new_ds.WriteMessage(DataLayoutMessage { chunked_prop });
    } else {
        len_t dataset_bytes = dataspace.MaxElements() * type.Size();

        offset_t data_alloc = object_.file->AllocateAtEOF(dataset_bytes);

        object_.file->io.SetPosition(data_alloc);

        new_ds.WriteMessage(DataLayoutMessage {
            ContiguousStorageProperty { .address = data_alloc, .size = dataset_bytes }
        });
    }

    new_ds.WriteMessage(ObjectModificationTimeMessage {
        .modification_time = cstd::chrono::system_clock::now(),
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

    auto insert_result = table_.InsertGroup(name_offset, node_alloc);
    if (!insert_result) {
        return cstd::unexpected(insert_result.error());
    }

    UpdateBTreePointer();

    return Dataset::New(new_ds);
}

hdf5::expected<Group> Group::OpenGroup(std::string_view group_name) const {
    auto object_result = Get(group_name);
    if (!object_result) {
        return cstd::unexpected(object_result.error());
    }

    if (const auto& object = *object_result) {
        return New(*object);
    }

    return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Group not found");
}

hdf5::expected<Group> Group::CreateGroup(std::string_view name) {
    auto exists_result = Get(name);
    if (!exists_result) {
        return cstd::unexpected(exists_result.error());
    }
    if (*exists_result) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "Group already exists");
    }

    auto name_offset_result = GetLocalHeap().WriteString(name, *object_.file);
    if (!name_offset_result) {
        return cstd::unexpected(name_offset_result.error());
    }
    offset_t name_offset = *name_offset_result;

    Object new_group_obj = Object::AllocateEmptyAtEOF(24 + 24, object_.file);

    BTreeNode::KValues k{
        .leaf = object_.file->superblock.group_leaf_node_k,
        .internal = object_.file->superblock.group_internal_node_k
    };

    auto [heap, heap_offset] = LocalHeap::AllocateNew(*object_.file, 64);

    auto empty_offset_result = heap.WriteString("", *object_.file);
    if (!empty_offset_result) {
        return cstd::unexpected(empty_offset_result.error());
    }
    offset_t empty_offset = *empty_offset_result;

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

    auto insert_result = table_.InsertGroup(name_offset, node_alloc);
    if (!insert_result) {
        return cstd::unexpected(insert_result.error());
    }

    UpdateBTreePointer();

    return New(new_group_obj);
}

hdf5::expected<cstd::optional<Object>> Group::Get(std::string_view name) const {
    auto sym_table_node_ptr_result = table_.Get(name);
    if (!sym_table_node_ptr_result) {
        return cstd::unexpected(sym_table_node_ptr_result.error());
    }
    cstd::optional<offset_t> sym_table_node_ptr = *sym_table_node_ptr_result;

    if (!sym_table_node_ptr) {
        return cstd::nullopt;
    }

    offset_t base_addr = object_.file->superblock.base_addr;

    object_.file->io.SetPosition(base_addr + *sym_table_node_ptr);
    auto symbol_table_node_result = object_.file->io.ReadComplex<SymbolTableNode>();
    if (!symbol_table_node_result) return cstd::unexpected(symbol_table_node_result.error());

    auto entry_addr_result = symbol_table_node_result->FindEntry(name, GetLocalHeap(), object_.file->io);
    if (!entry_addr_result) {
        return cstd::unexpected(entry_addr_result.error());
    }
    cstd::optional<offset_t> entry_addr = *entry_addr_result;

    if (!entry_addr) {
        return cstd::nullopt;
    }

    return Object::New(object_.file, base_addr + *entry_addr);
}

hdf5::expected<void> Group::Insert(std::string_view name, offset_t object_header_ptr) {
    auto name_offset_result = GetLocalHeap().WriteString(name, *object_.file);
    if (!name_offset_result) {
        return cstd::unexpected(name_offset_result.error());
    }
    offset_t name_offset = *name_offset_result;

    auto insert_result = table_.InsertGroup(name_offset, object_header_ptr);
    if (!insert_result) {
        return cstd::unexpected(insert_result.error());
    }

    UpdateBTreePointer();
    return {};
}

hdf5::expected<SymbolTableNode> Group::GetSymbolTableNode() const {
    auto root_result = table_.ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "BTree root is null");
    }

    BTreeNode& table = **root_result;

    if (!table.IsLeaf()) {
        return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "traversing tree not implemented");
    }

    const auto* entries = cstd::get_if<BTreeEntries<BTreeGroupNodeKey>>(&table.entries);

    if (!entries) {
        return hdf5::error(hdf5::HDF5ErrorCode::WrongNodeType, "Group table does not contain group node keys");
    }

    if (entries->child_pointers.size() != 1) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "nodes at level zero should only have one child pointer");
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

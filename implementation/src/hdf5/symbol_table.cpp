#include "symbol_table.h"

#include <stdexcept>

template<serde::Deserializer D>
hdf5::expected<cstd::optional<offset_t>> SymbolTableNode::FindEntry(hdf5::string_view name, const LocalHeap& heap, D& de) const {
    for (const auto& entry : entries) {
        // TODO(cuda_vector): this likely doesn't need to allocate if only used to check; might be a lifetime nightmare if made generally though
        auto entry_name = heap.ReadString(entry.link_name_offset, de);
        if (!entry_name) {
            return cstd::unexpected(entry_name.error());
        }

        if (*entry_name == name) {
            return entry.object_header_addr;
        }
    }

    return cstd::nullopt;
}

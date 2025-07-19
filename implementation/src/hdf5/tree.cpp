#include "tree.h"

void BTreeChunkedRawDataNodeKey::Serialize(Serializer& s) const {
    s.Write(chunk_size);

    for (const uint64_t offset: chunk_offset_in_dataset) {
        s.Write(offset);
    }

    s.Write<uint64_t>(0);
}

BTreeChunkedRawDataNodeKey BTreeChunkedRawDataNodeKey::Deserialize(Deserializer& de) {
    BTreeChunkedRawDataNodeKey key{};

    key.chunk_size = de.Read<uint32_t>();

    for (uint64_t offset; (offset = de.Read<uint64_t>()) != 0;) {
        key.chunk_offset_in_dataset.push_back(offset);
    }

    return key;
}

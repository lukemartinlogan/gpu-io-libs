#pragma once

#include "../defines.h"
#include "../serde.h"
#include "../utils/buffer.h"
#include <cuda/std/span>
#include <cuda/std/expected>

namespace kvhdf5 {

enum class BlobStoreError : uint8_t {
    NotExist,
    NotEnoughSpace
};

template<typename T>
concept RawBlobStore = requires(
    T store,
    cstd::span<const byte_t> key,
    cstd::span<const byte_t> value,
    cstd::span<byte_t> value_out) {

    { store.PutBlob(key, value) } -> cstd::same_as<bool>;

    { store.GetBlob(key, value_out) } -> cstd::same_as<cstd::expected<cstd::span<byte_t>, BlobStoreError>>;

    { store.DeleteBlob(key) } -> cstd::same_as<bool>;

    { store.Exists(key) } -> cstd::same_as<bool>;
};

template<typename Fn, typename V>
concept ValueSerializer = requires(Fn fn, serde::BufferReaderWriter& writer, const V& value) {
    { fn(writer, value) } -> cstd::same_as<void>;
};

template<typename Fn, typename V>
concept ValueDeserializer = requires(Fn fn, serde::BufferDeserializer& reader) {
    { fn(reader) } -> cstd::same_as<V>;
};

template<RawBlobStore BlobStoreImpl>
class TypedBlobStore {
    BlobStoreImpl* store_;

public:
    static constexpr size_t DefaultMaxValueSize = 1024;

    CROSS_FUN explicit TypedBlobStore(BlobStoreImpl* store) : store_(store) {}

    template<typename K, typename V>
        requires (serde::SerializePOD<K>::value && serde::SerializePOD<V>::value)
    CROSS_FUN bool PutBlob(const K& key, const V& value) {
        cstd::array<byte_t, sizeof(K)> key_buffer;
        serde::BufferReaderWriter key_writer(key_buffer);
        serde::Write(key_writer, key);

        cstd::array<byte_t, sizeof(V)> value_buffer;
        serde::BufferReaderWriter value_writer(value_buffer);
        serde::Write(value_writer, value);

        return store_->PutBlob(key_writer.GetWritten(), value_writer.GetWritten());
    }

    template<typename K, typename V>
        requires (serde::SerializePOD<K>::value && serde::SerializePOD<V>::value)
    CROSS_FUN cstd::expected<V, BlobStoreError> GetBlob(const K& key) {
        cstd::array<byte_t, sizeof(K)> key_buffer;
        serde::BufferReaderWriter key_writer(key_buffer);
        serde::Write(key_writer, key);

        cstd::array<byte_t, sizeof(V)> value_buffer;
        auto result = store_->GetBlob(key_writer.GetWritten(), value_buffer);
        if (!result) {
            return cstd::unexpected(result.error());
        }

        serde::BufferDeserializer reader(*result);
        return serde::Read<V>(reader);
    }

    template<typename K, typename V, ValueSerializer<V> SerializeFn, size_t MaxValueSize = DefaultMaxValueSize>
    requires serde::SerializePOD<K>::value
    CROSS_FUN bool PutBlob(const K& key, const V& value, SerializeFn&& value_serialize_fn) {
        cstd::array<byte_t, sizeof(K)> key_buffer;
        serde::BufferReaderWriter key_writer(key_buffer);
        serde::Write(key_writer, key);

        cstd::array<byte_t, MaxValueSize> value_buffer;
        serde::BufferReaderWriter value_writer(value_buffer);
        value_serialize_fn(value_writer, value);

        return store_->PutBlob(key_writer.GetWritten(), value_writer.GetWritten());
    }

    template<typename K, typename V, ValueDeserializer<V> DeserializeFn, size_t MaxValueSize = DefaultMaxValueSize>
    requires serde::SerializePOD<K>::value
    CROSS_FUN cstd::expected<V, BlobStoreError> GetBlob(const K& key, DeserializeFn&& value_deserialize_fn) {
        cstd::array<byte_t, sizeof(K)> key_buffer;
        serde::BufferReaderWriter key_writer(key_buffer);
        serde::Write(key_writer, key);

        cstd::array<byte_t, MaxValueSize> value_buffer;
        auto result = store_->GetBlob(key_writer.GetWritten(), value_buffer);
        if (!result) {
            return cstd::unexpected(result.error());
        }

        serde::BufferDeserializer reader(*result);
        return value_deserialize_fn(reader);
    }

    template<typename K>
    requires serde::SerializePOD<K>::value
    CROSS_FUN bool DeleteBlob(const K& key) {
        cstd::array<byte_t, sizeof(K)> key_buffer;
        serde::BufferReaderWriter key_writer(key_buffer);
        serde::Write(key_writer, key);

        return store_->DeleteBlob(key_writer.GetWritten());
    }


    template<typename K>
    requires serde::SerializePOD<K>::value
    CROSS_FUN bool Exists(const K& key) {
        cstd::array<byte_t, sizeof(K)> key_buffer;
        serde::BufferReaderWriter key_writer(key_buffer);
        serde::Write(key_writer, key);

        return store_->Exists(key_writer.GetWritten());
    }

    CROSS_FUN BlobStoreImpl* GetStore() { return store_; }
    CROSS_FUN const BlobStoreImpl* GetStore() const { return store_; }
};

} // namespace kvhdf5

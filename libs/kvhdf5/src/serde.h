#pragma once

#include "defines.h"
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/array>
#include <cuda/std/bit>

namespace serde {

template<typename T>
concept IsPOD = cstd::is_trivially_copyable_v<T> && cstd::is_standard_layout_v<T>;

template<typename T>
struct SerializePOD : cstd::false_type {};

template<typename S>
concept Serializer = requires(S&& s, cstd::span<const kvhdf5::byte_t> data) {
    { s.WriteBuffer(data) } -> cstd::same_as<void>;
};

template<typename D>
concept Deserializer = requires(D&& d, cstd::span<kvhdf5::byte_t> data) {
    { d.ReadBuffer(data) } -> cstd::same_as<void>;
};

template<typename S>
concept Seekable = requires(S&& s, size_t pos) {
    { s.GetPosition() } -> cstd::same_as<size_t>;
    { s.SetPosition(pos) } -> cstd::same_as<void>;
};

template<typename T, Serializer S> requires (SerializePOD<T>::value && IsPOD<T>)
CROSS_FUN void Write(S&& s, const T& data) {
    s.WriteBuffer(cstd::as_bytes(cstd::span(&data, 1)));
}

template<typename T, Deserializer D> requires (SerializePOD<T>::value && IsPOD<T>)
CROSS_FUN T Read(D&& d) {
    alignas(T) cstd::array<kvhdf5::byte_t, sizeof(T)> storage;
    d.ReadBuffer(cstd::span(storage.data(), storage.size()));
    return cstd::bit_cast<T>(storage);
}

template<Seekable S>
CROSS_FUN void Skip(S&& s, size_t count) {
    size_t start = s.GetPosition();
    s.SetPosition(start + count);
}

template<typename T, Seekable S>
CROSS_FUN void Skip(S&& s) {
    Skip(s, sizeof(T));
}

} // namespace serde

#define KVHDF5_AUTO_SERDE(T) \
    static_assert(serde::IsPOD<T>, #T " must be POD to enable auto serde"); \
    template<> struct serde::SerializePOD<T> : cstd::true_type {}

// opt in for built-in types
KVHDF5_AUTO_SERDE(uint8_t);
KVHDF5_AUTO_SERDE(uint16_t);
KVHDF5_AUTO_SERDE(uint32_t);
KVHDF5_AUTO_SERDE(uint64_t);
KVHDF5_AUTO_SERDE(int8_t);
KVHDF5_AUTO_SERDE(int16_t);
KVHDF5_AUTO_SERDE(int32_t);
KVHDF5_AUTO_SERDE(int64_t);
KVHDF5_AUTO_SERDE(float);
KVHDF5_AUTO_SERDE(double);
KVHDF5_AUTO_SERDE(bool);
KVHDF5_AUTO_SERDE(char);

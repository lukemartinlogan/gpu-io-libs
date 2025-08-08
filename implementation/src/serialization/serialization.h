#pragma once

#include <span>
#include <array>
#include "../hdf5/types.h"

template<typename T>
constexpr bool is_trivially_serializable_v = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

class Serializer {
public:
    virtual ~Serializer() = default;

    virtual bool WriteBuffer(std::span<const byte_t> data) = 0;

    template<typename T>
    void WriteRaw(const T& data) {
        static_assert(is_trivially_serializable_v<T>, "not safely serializable");

        [[maybe_unused]]
        bool _ = WriteBuffer(std::as_bytes(std::span(&data, 1)));
    }

    template<typename T>
    void WriteComplex(const T& data) {
        data.Serialize(*this);
    }

    template<typename T>
    void Write(const T& data) {
        if constexpr (is_trivially_serializable_v<T>) {
            WriteRaw(data);
        } else {
            WriteComplex(data);
        }
    }
};

class Deserializer {
public:
    virtual ~Deserializer() = default;

    virtual bool ReadBuffer(std::span<byte_t> out) = 0;

    [[nodiscard]] virtual offset_t GetPosition() = 0;
    virtual void SetPosition(offset_t offset) = 0;

    template<typename T>
    T ReadRaw() {
        static_assert(is_trivially_serializable_v<T>, "not safely deserializable");

        T out;
        ReadBuffer(std::as_writable_bytes(std::span(&out, 1)));
        return out;
    }

    template<typename T>
    T ReadComplex() {
        return T::Deserialize(*this);
    }

    template<typename T>
    T Read() {
        if constexpr (is_trivially_serializable_v<T>) {
            return ReadRaw<T>();
        } else {
            return ReadComplex<T>();
        }
    }

    template<size_t count>
    void Skip() {
        std::array<byte_t, count> buf{};
        auto _ = ReadBuffer(buf);
    }

    template<typename T>
    void Skip() {
        Skip<sizeof(T)>();
    }
};
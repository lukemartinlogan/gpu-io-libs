#pragma once

#include <span>
#include "../hdf5/types.h"

namespace serde {
    // (quirk with C++ concepts)
    struct NullSerializer {
        __device__ __host__
        NullSerializer() = delete;

        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        __device__ __host__
        void WriteBuffer(cstd::span<const byte_t>) {
            ASSERT(false, "NullSerializer shouldn't be called");
        }
    };

    struct NullDeserializer {
        __device__ __host__
        NullDeserializer() = delete;

        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        __device__ __host__
        void ReadBuffer(cstd::span<byte_t>) {
            ASSERT(false, "NullDeserializer shouldn't be called");
        }

        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        __device__ __host__
        offset_t GetPosition() {
            ASSERT(false, "NullDeserializer shouldn't be called");
            return 0;
        }
        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        __device__ __host__
        void SetPosition(offset_t) {
            ASSERT(false, "NullDeserializer shouldn't be called");
        }
    };

    // -- CONCEPTS
    template<typename S>
    concept Serializer = requires(S&& s, cstd::span<const byte_t> data) {
        { s.WriteBuffer(data) } -> std::same_as<void>;
    };

    template<typename S>
    concept Seekable = requires(S&& s, offset_t pos) {
        { s.GetPosition() } -> std::same_as<offset_t>;
        { s.SetPosition(pos) } -> std::same_as<void>;
    };

    template<typename D>
    concept Deserializer = Seekable<D> && requires(D&& d, cstd::span<byte_t> data) {
        { d.ReadBuffer(data) } -> std::same_as<void>;
    };

    // -- DATA TYPES --
    template<typename T>
    concept NonTriviallySerializable = requires(const T& t, NullSerializer& s) { { t.Serialize(s) } -> std::same_as<void>; }
        || requires(NullDeserializer& d) { { T::Deserialize(d) } -> std::same_as<T>; }
        || requires(NullDeserializer& d) { { T::Deserialize(d) } -> std::same_as<hdf5::expected<T>>; };

    template<typename T>
    concept TriviallySerializable = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && !NonTriviallySerializable<T>;

    // -- SERIALIZE FUNCTIONS --

    template<TriviallySerializable T, Serializer S>
    __device__ __host__
    void Write(S&& s, const T& data) {
        s.WriteBuffer(cstd::as_bytes(cstd::span(&data, 1)));
    }

    template<NonTriviallySerializable T, Serializer S>
    __device__ __host__
    void Write(S&& s, const T& data) {
        data.Serialize(s);
    }

    // -- DESERIALIZE FUNCTIONS --

    template<TriviallySerializable T, Deserializer D>
    __device__ __host__
    T Read(D&& d) {
        T out;
        d.ReadBuffer(cstd::as_writable_bytes(cstd::span(&out, 1)));
        return out;
    }

    template<NonTriviallySerializable T, Deserializer D>
    __device__ __host__
    auto Read(D&& d) {
        return T::Deserialize(d);
    }

    // -- SEEKABLE FUNCTIONS --

    template<Seekable S>
    __device__ __host__
    void Skip(S&& s, offset_t count) {
        offset_t start = s.GetPosition();
        s.SetPosition(start + count);
    }

    template<typename T, Seekable S>
    __device__ __host__
    void Skip(S&& s) {
        Skip(s, sizeof(T));
    }
}
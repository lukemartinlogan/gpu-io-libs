#pragma once

#include <span>
#include <array>
#include "../hdf5/types.h"

template<typename T>
constexpr bool is_trivially_serializable_v = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

class VirtualSerializer {
public:
    virtual ~VirtualSerializer() = default;

    virtual bool WriteBuffer(cstd::span<const byte_t> data) = 0;

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

class VirtualDeserializer {
public:
    virtual ~VirtualDeserializer() = default;

    virtual bool ReadBuffer(cstd::span<byte_t> out) = 0;

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
    auto ReadComplex() {
        return T::Deserialize(*this);
    }

    template<typename T>
    auto Read() {
        if constexpr (is_trivially_serializable_v<T>) {
            return ReadRaw<T>();
        } else {
            return ReadComplex<T>();
        }
    }

    template<size_t count>
    void Skip() {
        cstd::array<byte_t, count> buf{};
        auto _ = ReadBuffer(buf);
    }

    template<typename T>
    void Skip() {
        Skip<sizeof(T)>();
    }
};

class VirtualReaderWriter : public VirtualSerializer, public VirtualDeserializer {
public:
    // bool WriteBuffer(std::span<const byte_t> data) override = 0;
    bool WriteBuffer(cstd::span<const byte_t> data) override = 0;
    bool ReadBuffer(cstd::span<byte_t> out) override = 0;
    [[nodiscard]] offset_t GetPosition() override = 0;
    void SetPosition(offset_t offset) override = 0;
    ~VirtualReaderWriter() override = default;
};

namespace serde {
    // (quirk with C++ concepts)
    struct NullSerializer {
        NullSerializer() = delete;

        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        void WriteBuffer(cstd::span<const byte_t>) {
            ASSERT(false, "NullSerializer shouldn't be called");
        }
    };

    struct NullDeserializer {
        NullDeserializer() = delete;

        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        void ReadBuffer(cstd::span<byte_t>) {
            ASSERT(false, "NullDeserializer shouldn't be called");
        }

        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        offset_t GetPosition() {
            ASSERT(false, "NullDeserializer shouldn't be called");
            return 0;
        }
        // ReSharper disable once CppMemberFunctionMayBeStatic
        // NOLINTNEXTLINE(*-convert-member-functions-to-static)
        void SetPosition(offset_t) {
            ASSERT(false, "NullDeserializer shouldn't be called");
        }
    };

    // -- CONCEPTS
    template<typename S>
    concept Serializer = requires(S& s, cstd::span<const byte_t> data) {
        { s.WriteBuffer(data) } -> std::same_as<void>;
    };

    template<typename S>
    concept Seekable = requires(S& s, offset_t pos) {
        { s.GetPosition() } -> std::same_as<offset_t>;
        { s.SetPosition(pos) } -> std::same_as<void>;
    };

    template<typename D>
    concept Deserializer = Seekable<D> && requires(D& d, cstd::span<byte_t> data) {
        { d.ReadBuffer(data) } -> std::same_as<void>;
    };

    // -- DATA TYPES --
    template<typename T>
    concept NonTriviallySerializable = requires(const T& t, NullSerializer& s) {
        { t.Serialize(s) } -> std::same_as<void>;
    } && (
        requires(NullDeserializer& d) { { T::Deserialize(d) } -> std::same_as<void>; } ||
        requires(NullDeserializer& d) { { T::Deserialize(d) } -> std::same_as<hdf5::expected<void>>; }
    );

    template<typename T>
    concept TriviallySerializable = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && !NonTriviallySerializable<T>;

    // -- SERIALIZE FUNCTIONS --

    template<Serializer S, TriviallySerializable T>
    void Write(S& s, const T& data) {
        s.WriteBuffer(std::as_bytes(std::span(&data, 1)));
    }

    template<Serializer S, NonTriviallySerializable T>
    void Write(S& s, const T& data) {
        data.Serialize(s);
    }

    // -- DESERIALIZE FUNCTIONS --

    template<Deserializer D, TriviallySerializable T>
    T Read(D& d) {
        T out;
        d.ReadBuffer(std::as_writable_bytes(std::span(&out, 1)));
        return out;
    }

    template<Deserializer D, NonTriviallySerializable T>
    auto Read(D& d) {
        return T::Deserialize(d);
    }

    // -- SEEKABLE FUNCTIONS --

    template<Seekable S>
    void Skip(S& s, offset_t count) {
        offset_t start = s.GetPosition();
        s.SetPosition(start + count);
    }

    template<Seekable S, typename T>
    void Skip(S& s) {
        Skip(s, sizeof(T));
    }
}
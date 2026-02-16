#include <catch2/catch_test_macros.hpp>
#include <serde.h>
#include <utils/buffer.h>
#include <cuda/std/array>

using kvhdf5::byte_t;

// Test POD struct
struct TestPOD {
    uint32_t a;
    uint64_t b;
    float c;

    constexpr bool operator==(const TestPOD& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

// Opt-in TestPOD for serialization
KVHDF5_AUTO_SERDE(TestPOD);

TEST_CASE("Serde - Primitive types round-trip", "[serde]") {
    cstd::array<byte_t, 1024> buffer{};
    serde::BufferReaderWriter rw(cstd::span(buffer.data(), buffer.size()));

    SECTION("uint8_t") {
        uint8_t val = 42;
        serde::Write(rw, val);
        rw.SetPosition(0);
        auto result = serde::Read<uint8_t>(rw);
        CHECK(result == val);
    }

    SECTION("uint16_t") {
        uint16_t val = 12345;
        serde::Write(rw, val);
        rw.SetPosition(0);
        auto result = serde::Read<uint16_t>(rw);
        CHECK(result == val);
    }

    SECTION("uint32_t") {
        uint32_t val = 0xDEADBEEF;
        serde::Write(rw, val);
        rw.SetPosition(0);
        auto result = serde::Read<uint32_t>(rw);
        CHECK(result == val);
    }

    SECTION("uint64_t") {
        uint64_t val = 0xCAFEBABEDEADBEEF;
        serde::Write(rw, val);
        rw.SetPosition(0);
        auto result = serde::Read<uint64_t>(rw);
        CHECK(result == val);
    }

    SECTION("float") {
        float val = 3.14159f;
        serde::Write(rw, val);
        rw.SetPosition(0);
        auto result = serde::Read<float>(rw);
        CHECK(result == val);
    }

    SECTION("double") {
        double val = 2.718281828;
        serde::Write(rw, val);
        rw.SetPosition(0);
        auto result = serde::Read<double>(rw);
        CHECK(result == val);
    }

    SECTION("bool") {
        bool val_true = true;
        bool val_false = false;
        serde::Write(rw, val_true);
        serde::Write(rw, val_false);
        rw.SetPosition(0);
        auto result_true = serde::Read<bool>(rw);
        auto result_false = serde::Read<bool>(rw);
        CHECK(result_true == true);
        CHECK(result_false == false);
    }
}

TEST_CASE("Serde - Custom POD struct round-trip", "[serde]") {
    cstd::array<byte_t, 1024> buffer{};
    serde::BufferReaderWriter rw(cstd::span(buffer.data(), buffer.size()));

    TestPOD original{123, 456789, 3.14f};
    Write(rw, original);

    rw.SetPosition(0);
    auto result = serde::Read<TestPOD>(rw);

    CHECK(result == original);
    CHECK(result.a == 123);
    CHECK(result.b == 456789);
    CHECK(result.c == 3.14f);
}

TEST_CASE("Serde - Multiple values sequential", "[serde]") {
    cstd::array<byte_t, 1024> buffer{};
    serde::BufferReaderWriter rw(cstd::span(buffer.data(), buffer.size()));

    // Write multiple values
    Write(rw, uint32_t(100));
    Write(rw, uint64_t(200));
    Write(rw, float(3.14f));

    // Read back
    rw.SetPosition(0);
    CHECK(serde::Read<uint32_t>(rw) == 100);
    CHECK(serde::Read<uint64_t>(rw) == 200);
    CHECK(serde::Read<float>(rw) == 3.14f);
}

TEST_CASE("serde::BufferDeserializer - Read operations", "[buffer][deserializer]") {
    // Prepare test data
    cstd::array<byte_t, 16> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<byte_t>(i);
    }

    serde::BufferDeserializer deser(cstd::span<const byte_t>(data.data(), data.size()));

    SECTION("ReadBuffer works correctly") {
        cstd::array<byte_t, 4> out{};
        deser.ReadBuffer(cstd::span(out.data(), out.size()));

        CHECK(static_cast<uint8_t>(out[0]) == 0);
        CHECK(static_cast<uint8_t>(out[1]) == 1);
        CHECK(static_cast<uint8_t>(out[2]) == 2);
        CHECK(static_cast<uint8_t>(out[3]) == 3);
        CHECK(deser.GetPosition() == 4);
    }

    SECTION("GetPosition and SetPosition") {
        CHECK(deser.GetPosition() == 0);

        cstd::array<byte_t, 4> out{};
        deser.ReadBuffer(cstd::span(out.data(), out.size()));
        CHECK(deser.GetPosition() == 4);

        deser.SetPosition(8);
        CHECK(deser.GetPosition() == 8);

        deser.ReadBuffer(cstd::span(out.data(), out.size()));
        CHECK(static_cast<uint8_t>(out[0]) == 8);
        CHECK(deser.GetPosition() == 12);
    }

    SECTION("IsExhausted and Remaining") {
        CHECK(!deser.IsExhausted());
        CHECK(deser.Remaining() == 16);

        cstd::array<byte_t, 16> all{};
        deser.ReadBuffer(cstd::span(all.data(), all.size()));

        CHECK(deser.IsExhausted());
        CHECK(deser.Remaining() == 0);
    }
}

TEST_CASE("serde::BufferReaderWriter - Write and read operations", "[buffer][readerwriter]") {
    cstd::array<byte_t, 64> buffer{};
    serde::BufferReaderWriter rw(cstd::span(buffer.data(), buffer.size()));

    SECTION("WriteBuffer works correctly") {
        cstd::array<byte_t, 4> data{byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};
        rw.WriteBuffer(cstd::span<const byte_t>(data.data(), data.size()));

        CHECK(rw.GetPosition() == 4);

        // Read back
        rw.SetPosition(0);
        cstd::array<byte_t, 4> out{};
        rw.ReadBuffer(cstd::span(out.data(), out.size()));

        CHECK(static_cast<uint8_t>(out[0]) == 10);
        CHECK(static_cast<uint8_t>(out[1]) == 20);
        CHECK(static_cast<uint8_t>(out[2]) == 30);
        CHECK(static_cast<uint8_t>(out[3]) == 40);
    }

    SECTION("GetWritten returns written data") {
        cstd::array<byte_t, 8> data{};
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<byte_t>(i * 10);
        }

        rw.WriteBuffer(cstd::span<const byte_t>(data.data(), data.size()));

        auto written = rw.GetWritten();
        CHECK(written.size() == 8);
        CHECK(static_cast<uint8_t>(written[0]) == 0);
        CHECK(static_cast<uint8_t>(written[7]) == 70);
    }

    SECTION("Position management") {
        CHECK(rw.GetPosition() == 0);

        cstd::array<byte_t, 4> data{byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        rw.WriteBuffer(cstd::span<const byte_t>(data.data(), data.size()));
        CHECK(rw.GetPosition() == 4);

        rw.SetPosition(2);
        CHECK(rw.GetPosition() == 2);

        cstd::array<byte_t, 2> out{};
        rw.ReadBuffer(cstd::span(out.data(), out.size()));
        CHECK(static_cast<uint8_t>(out[0]) == 3);
        CHECK(static_cast<uint8_t>(out[1]) == 4);
    }

    SECTION("IsExhausted and Remaining") {
        CHECK(!rw.IsExhausted());
        CHECK(rw.Remaining() == 64);

        cstd::array<byte_t, 32> data{};
        rw.WriteBuffer(cstd::span<const byte_t>(data.data(), data.size()));

        CHECK(!rw.IsExhausted());
        CHECK(rw.Remaining() == 32);
        CHECK(rw.GetPosition() == 32);
    }
}

TEST_CASE("Seekable - Skip functions", "[serde][seekable]") {
    cstd::array<byte_t, 64> buffer{};
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<byte_t>(i);
    }

    serde::BufferDeserializer deser(cstd::span<const byte_t>(buffer.data(), buffer.size()));

    SECTION("Skip by count") {
        CHECK(deser.GetPosition() == 0);

        serde::Skip(deser, 10);
        CHECK(deser.GetPosition() == 10);

        cstd::array<byte_t, 1> out{};
        deser.ReadBuffer(cstd::span(out.data(), out.size()));
        CHECK(static_cast<uint8_t>(out[0]) == 10);
    }

    SECTION("Skip by type size") {
        CHECK(deser.GetPosition() == 0);

        serde::Skip<uint64_t>(deser);
        CHECK(deser.GetPosition() == 8);

        serde::Skip<uint32_t>(deser);
        CHECK(deser.GetPosition() == 12);
    }
}

TEST_CASE("Serde - Endianness consistency", "[serde]") {
    cstd::array<byte_t, 1024> buffer{};
    serde::BufferReaderWriter rw(cstd::span(buffer.data(), buffer.size()));

    // Write known value
    uint32_t val = 0x12345678;
    Write(rw, val);

    // Check bytes (will be little-endian on most systems)
    auto written = rw.GetWritten();
    CHECK(written.size() == 4);

    // Read back - should match regardless of byte order
    rw.SetPosition(0);
    auto result = serde::Read<uint32_t>(rw);
    CHECK(result == val);
}

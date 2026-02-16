#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/types.h>
#include <utils/buffer.h>
#include <cuda/std/array>

using namespace kvhdf5;
using serde::BufferReaderWriter;

TEST_CASE("ObjectId construction and validity", "[types]") {
    SECTION("Default construction creates invalid ID") {
        ObjectId id;
        REQUIRE_FALSE(id.IsValid());
        REQUIRE(id.id == 0);
    }

    SECTION("Explicit construction with non-zero creates valid ID") {
        ObjectId id(42);
        REQUIRE(id.IsValid());
        REQUIRE(id.id == 42);
    }

    SECTION("Explicit construction with zero creates invalid ID") {
        ObjectId id(0);
        REQUIRE_FALSE(id.IsValid());
    }
}

TEST_CASE("ObjectId comparison operators", "[types]") {
    ObjectId id1(10);
    ObjectId id2(20);
    ObjectId id3(10);

    SECTION("Equality") {
        REQUIRE(id1 == id3);
        REQUIRE_FALSE(id1 == id2);
    }

    SECTION("Ordering") {
        REQUIRE(id1 < id2);
        REQUIRE(id2 > id1);
        REQUIRE(id1 <= id3);
        REQUIRE(id1 >= id3);
    }
}

TEST_CASE("GroupId type safety", "[types]") {
    GroupId gid(100);
    DatasetId did(100);

    SECTION("GroupId wraps ObjectId correctly") {
        REQUIRE(gid.IsValid());
        REQUIRE(gid.Id() == 100);
    }

    SECTION("GroupId and DatasetId are distinct types") {
        // This should not compile if uncommented (type safety check):
        // gid = did;  // Should be a compile error

        // But their underlying values can be equal
        REQUIRE(gid.Id() == did.Id());
    }
}

TEST_CASE("DatasetId construction and comparison", "[types]") {
    DatasetId did1(50);
    DatasetId did2(50);
    DatasetId did3(60);

    SECTION("Equality works") {
        REQUIRE(did1 == did2);
        REQUIRE_FALSE(did1 == did3);
    }

    SECTION("Ordering works") {
        REQUIRE(did1 < did3);
        REQUIRE(did3 > did1);
    }
}

TEST_CASE("DatatypeId construction and validity", "[types]") {
    DatatypeId tid(ObjectId(123));

    SECTION("Construction from ObjectId") {
        REQUIRE(tid.IsValid());
        REQUIRE(tid.Id() == 123);
    }

    SECTION("Invalid DatatypeId") {
        DatatypeId invalid;
        REQUIRE_FALSE(invalid.IsValid());
    }
}

TEST_CASE("ChunkKey construction", "[types]") {
    DatasetId ds(42);
    uint64_t coords[] = {1, 2, 3};

    SECTION("Construction with coordinates") {
        ChunkKey key(ds, coords);
        REQUIRE(key.dataset == ds);
        REQUIRE(key.ndims() == 3);
        REQUIRE(key.coords[0] == 1);
        REQUIRE(key.coords[1] == 2);
        REQUIRE(key.coords[2] == 3);
    }

    SECTION("Default construction") {
        ChunkKey key;
        REQUIRE_FALSE(key.dataset.IsValid());
        REQUIRE(key.ndims() == 0);
    }

    SECTION("Construction with MAX_DIMS coordinates") {
        uint64_t max_coords[MAX_DIMS] = {0, 1, 2, 3, 4, 5, 6, 7};
        ChunkKey key(ds, max_coords);
        REQUIRE(key.dataset == ds);
        REQUIRE(key.ndims() == MAX_DIMS);
        for (size_t i = 0; i < MAX_DIMS; ++i) {
            REQUIRE(key.coords[i] == i);
        }
    }

    // Note: Exceeding MAX_DIMS triggers KVHDF5_ASSERT and aborts.
    // This cannot be tested in standard unit tests.
}

TEST_CASE("ChunkKey lexicographic ordering", "[types]") {
    DatasetId ds1(10);
    DatasetId ds2(20);

    SECTION("Different datasets") {
        uint64_t coords[] = {0, 0};
        ChunkKey key1(ds1, coords);
        ChunkKey key2(ds2, coords);

        REQUIRE(key1 < key2);
        REQUIRE(key2 > key1);
    }

    SECTION("Same dataset, different ndims") {
        uint64_t coords1[] = {5};
        uint64_t coords2[] = {5, 0};
        ChunkKey key1(ds1, coords1);
        ChunkKey key2(ds1, coords2);

        REQUIRE(key1 < key2);
    }

    SECTION("Same dataset and ndims, different coords") {
        uint64_t coords1[] = {1, 2, 3};
        uint64_t coords2[] = {1, 2, 4};
        ChunkKey key1(ds1, coords1);
        ChunkKey key2(ds1, coords2);

        REQUIRE(key1 < key2);
    }

    SECTION("Identical keys are equal") {
        uint64_t coords[] = {7, 8, 9};
        ChunkKey key1(ds1, coords);
        ChunkKey key2(ds1, coords);

        REQUIRE(key1 == key2);
        REQUIRE_FALSE(key1 < key2);
        REQUIRE_FALSE(key2 < key1);
    }
}

TEST_CASE("ChunkKey multi-dimensional coordinate ordering", "[types]") {
    DatasetId ds(1);

    SECTION("2D coordinates") {
        uint64_t c1[] = {0, 0};
        uint64_t c2[] = {0, 1};
        uint64_t c3[] = {1, 0};
        uint64_t c4[] = {1, 1};

        ChunkKey k1(ds, c1);
        ChunkKey k2(ds, c2);
        ChunkKey k3(ds, c3);
        ChunkKey k4(ds, c4);

        REQUIRE(k1 < k2);
        REQUIRE(k2 < k3);
        REQUIRE(k3 < k4);
    }

    SECTION("3D coordinates with middle dimension differing") {
        uint64_t c1[] = {5, 3, 7};
        uint64_t c2[] = {5, 4, 2};  // Middle coord larger wins even if last is smaller

        ChunkKey k1(ds, c1);
        ChunkKey k2(ds, c2);

        REQUIRE(k1 < k2);
    }
}

TEST_CASE("ObjectId serialization", "[types]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize ObjectId") {
        ObjectId original(12345);

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<ObjectId>(rw);
        REQUIRE(result == original);
    }
}

TEST_CASE("GroupId serialization", "[types]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize GroupId") {
        GroupId original(999);

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<GroupId>(rw);
        REQUIRE(result == original);
    }
}

TEST_CASE("DatasetId serialization", "[types]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize DatasetId") {
        DatasetId original(777);

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<DatasetId>(rw);
        REQUIRE(result == original);
    }
}

TEST_CASE("DatatypeId serialization", "[types]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize DatatypeId") {
        DatatypeId original(555);

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<DatatypeId>(rw);
        REQUIRE(result == original);
    }
}

TEST_CASE("ChunkKey serialization", "[types]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize ChunkKey") {
        DatasetId ds(42);
        uint64_t coords[] = {10, 20, 30};
        ChunkKey original(ds, coords);

        serde::Write(rw, original);

        rw.Reset();
        auto deserialized = serde::Read<ChunkKey>(rw);
        REQUIRE(deserialized == original);
    }

    SECTION("ChunkKey with max dimensions") {
        DatasetId ds(100);
        uint64_t coords[] = {1, 2, 3, 4, 5, 6, 7, 8};
        ChunkKey original(ds, coords);

        serde::Write(rw, original);

        rw.Reset();
        auto deserialized = serde::Read<ChunkKey>(rw);
        REQUIRE(deserialized == original);
    }
}

#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/memory_blob_store.h>
#include <kvhdf5/blob_store.h>
#include <kvhdf5/types.h>
#include <kvhdf5/group.h>
#include <cuda/std/algorithm>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("InMemoryBlobStore - Raw API", "[blob_store]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    InMemoryBlobStore store(fixture.allocator);

    SECTION("Basic put and get") {
        // Create key and value
        std::array<byte_t, 4> key_data = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 8> value_data = {
            byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40},
            byte_t{50}, byte_t{60}, byte_t{70}, byte_t{80}
        };

        // Put blob
        REQUIRE(store.PutBlob(key_data, value_data));
        REQUIRE(store.Size() == 1);

        // Get blob
        std::array<byte_t, 8> output;
        auto result = store.GetBlob(key_data, output);

        REQUIRE(result.has_value());
        REQUIRE(result->size() == 8);

        // Verify data matches
        REQUIRE(cstd::equal(result->begin(), result->end(), value_data.begin()));
    }

    SECTION("Overwrite existing key") {
        std::array<byte_t, 4> key_data = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 4> value1 = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};
        std::array<byte_t, 4> value2 = {byte_t{99}, byte_t{88}, byte_t{77}, byte_t{66}};

        // Put initial value
        REQUIRE(store.PutBlob(key_data, value1));
        REQUIRE(store.Size() == 1);

        // Overwrite
        REQUIRE(store.PutBlob(key_data, value2));
        REQUIRE(store.Size() == 1);  // Should still be 1 entry

        // Verify new value
        std::array<byte_t, 4> output;
        auto result = store.GetBlob(key_data, output);

        REQUIRE(result.has_value());
        REQUIRE(cstd::equal(result->begin(), result->end(), value2.begin()));
    }

    SECTION("Get with insufficient buffer") {
        std::array<byte_t, 4> key_data = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 8> value_data = {
            byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40},
            byte_t{50}, byte_t{60}, byte_t{70}, byte_t{80}
        };

        REQUIRE(store.PutBlob(key_data, value_data));

        // Try to get with too small buffer
        std::array<byte_t, 4> small_output;
        auto result = store.GetBlob(key_data, small_output);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotEnoughSpace);
    }

    SECTION("Get non-existent key") {
        std::array<byte_t, 4> key_data = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 8> output;

        auto result = store.GetBlob(key_data, output);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Delete blob") {
        std::array<byte_t, 4> key_data = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 4> value_data = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};

        // Put blob
        REQUIRE(store.PutBlob(key_data, value_data));
        REQUIRE(store.Size() == 1);
        REQUIRE(store.Exists(key_data));

        // Delete blob
        REQUIRE(store.DeleteBlob(key_data));
        REQUIRE(store.Size() == 0);
        REQUIRE_FALSE(store.Exists(key_data));

        // Delete non-existent key should return false
        REQUIRE_FALSE(store.DeleteBlob(key_data));
    }

    SECTION("Exists check") {
        std::array<byte_t, 4> key1 = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 4> key2 = {byte_t{5}, byte_t{6}, byte_t{7}, byte_t{8}};
        std::array<byte_t, 4> value = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};

        REQUIRE_FALSE(store.Exists(key1));

        store.PutBlob(key1, value);

        REQUIRE(store.Exists(key1));
        REQUIRE_FALSE(store.Exists(key2));
    }

    SECTION("Multiple entries") {
        std::array<byte_t, 4> key1 = {byte_t{1}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> key2 = {byte_t{2}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> key3 = {byte_t{3}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> value = {byte_t{99}, byte_t{99}, byte_t{99}, byte_t{99}};

        store.PutBlob(key1, value);
        store.PutBlob(key2, value);
        store.PutBlob(key3, value);

        REQUIRE(store.Size() == 3);
        REQUIRE(store.Exists(key1));
        REQUIRE(store.Exists(key2));
        REQUIRE(store.Exists(key3));
    }

    SECTION("Clear") {
        std::array<byte_t, 4> key1 = {byte_t{1}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> key2 = {byte_t{2}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> value = {byte_t{99}, byte_t{99}, byte_t{99}, byte_t{99}};

        store.PutBlob(key1, value);
        store.PutBlob(key2, value);
        REQUIRE(store.Size() == 2);

        store.Clear();
        REQUIRE(store.Size() == 0);
        REQUIRE_FALSE(store.Exists(key1));
        REQUIRE_FALSE(store.Exists(key2));
    }

    SECTION("Empty key and value") {
        std::array<byte_t, 0> empty_key;
        std::array<byte_t, 0> empty_value;

        REQUIRE(store.PutBlob(empty_key, empty_value));
        REQUIRE(store.Size() == 1);

        std::array<byte_t, 0> output;
        auto result = store.GetBlob(empty_key, output);
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 0);
    }
}

TEST_CASE("BlobStore - POD types", "[blob_store][typed]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    InMemoryBlobStore raw_store(fixture.allocator);
    BlobStore<InMemoryBlobStore> store(&raw_store);

    SECTION("Put and get POD key-value") {
        ObjectId key(42);
        uint64_t value = 12345;

        REQUIRE(store.PutBlob(key, value));

        auto result = store.GetBlob<ObjectId, uint64_t>(key);
        REQUIRE(result.has_value());
        REQUIRE(*result == value);
    }

    SECTION("Get non-existent POD key") {
        ObjectId key(99);

        auto result = store.GetBlob<ObjectId, uint64_t>(key);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Overwrite POD value") {
        GroupId key(100);
        uint32_t value1 = 111;
        uint32_t value2 = 222;

        store.PutBlob(key, value1);
        store.PutBlob(key, value2);

        auto result = store.GetBlob<GroupId, uint32_t>(key);
        REQUIRE(result.has_value());
        REQUIRE(*result == value2);
    }

    SECTION("Multiple POD entries") {
        store.PutBlob(ObjectId(1), uint64_t{100});
        store.PutBlob(ObjectId(2), uint64_t{200});
        store.PutBlob(ObjectId(3), uint64_t{300});

        REQUIRE(raw_store.Size() == 3);

        auto r1 = store.GetBlob<ObjectId, uint64_t>(ObjectId(1));
        auto r2 = store.GetBlob<ObjectId, uint64_t>(ObjectId(2));
        auto r3 = store.GetBlob<ObjectId, uint64_t>(ObjectId(3));

        REQUIRE(r1.has_value());
        REQUIRE(r2.has_value());
        REQUIRE(r3.has_value());

        REQUIRE(*r1 == 100);
        REQUIRE(*r2 == 200);
        REQUIRE(*r3 == 300);
    }

    SECTION("Delete POD key") {
        DatasetId key(50);
        uint64_t value = 999;

        store.PutBlob(key, value);
        REQUIRE(store.Exists(key));

        REQUIRE(store.DeleteBlob(key));
        REQUIRE_FALSE(store.Exists(key));
    }

    SECTION("Exists check for POD key") {
        DatatypeId key(25);

        REQUIRE_FALSE(store.Exists(key));

        store.PutBlob(key, uint32_t{555});

        REQUIRE(store.Exists(key));
    }
}

TEST_CASE("BlobStore - Custom value serialization", "[blob_store][typed]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    InMemoryBlobStore raw_store(fixture.allocator);
    BlobStore<InMemoryBlobStore> store(&raw_store);
    Context ctx(fixture.allocator);

    SECTION("Put and get GroupMetadata") {
        GroupId key(1);
        GroupMetadata metadata{key, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};

        // Add a child
        metadata.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(10)), "test_dataset"));

        // Put with custom serialization
        bool put_result = store.PutBlob(
            key,
            metadata,
            [](auto& s, const auto& v) { v.Serialize(s); }
        );
        REQUIRE(put_result);

        // Get with custom deserialization
        auto result = store.GetBlob<GroupId, GroupMetadata>(
            key,
            [&ctx](auto& d) { return GroupMetadata::Deserialize(d, ctx); }
        );

        REQUIRE(result.has_value());
        REQUIRE(result->id == key);
        REQUIRE(result->children.size() == 1);
        REQUIRE(result->children[0].kind == ChildKind::Dataset);
        REQUIRE(result->children[0].object_id == ObjectId(10));
        REQUIRE(result->children[0].name == "test_dataset");
    }

    SECTION("Multiple GroupMetadata entries") {
        GroupId key1(1);
        GroupId key2(2);

        GroupMetadata meta1{key1, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta1.children.push_back(GroupEntry::NewGroup(GroupId(ObjectId(100)), "child1"));

        GroupMetadata meta2{key2, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta2.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(200)), "child2"));

        // Put both
        store.PutBlob(key1, meta1, [](auto& s, const auto& v) { v.Serialize(s); });
        store.PutBlob(key2, meta2, [](auto& s, const auto& v) { v.Serialize(s); });

        // Get both
        auto r1 = store.GetBlob<GroupId, GroupMetadata>(
            key1, [&ctx](auto& d) { return GroupMetadata::Deserialize(d, ctx); });
        auto r2 = store.GetBlob<GroupId, GroupMetadata>(
            key2, [&ctx](auto& d) { return GroupMetadata::Deserialize(d, ctx); });

        REQUIRE(r1.has_value());
        REQUIRE(r2.has_value());
        REQUIRE(r1->id == key1);
        REQUIRE(r2->id == key2);
        REQUIRE(r1->children.size() == 1);
        REQUIRE(r2->children.size() == 1);
    }

    SECTION("Overwrite with custom serialization") {
        GroupId key(5);

        // First metadata
        GroupMetadata meta1{key, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta1.children.push_back(GroupEntry::NewGroup(GroupId(ObjectId(10)), "old"));

        // Second metadata
        GroupMetadata meta2{key, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta2.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(20)), "new"));

        // Put first
        store.PutBlob(key, meta1, [](auto& s, const auto& v) { v.Serialize(s); });

        // Overwrite with second
        store.PutBlob(key, meta2, [](auto& s, const auto& v) { v.Serialize(s); });

        // Should only have 1 entry
        REQUIRE(raw_store.Size() == 1);

        // Get should return second metadata
        auto result = store.GetBlob<GroupId, GroupMetadata>(
            key, [&ctx](auto& d) { return GroupMetadata::Deserialize(d, ctx); });

        REQUIRE(result.has_value());
        REQUIRE(result->children.size() == 1);
        REQUIRE(result->children[0].name == "new");
    }

}

TEST_CASE("BlobStore - ChunkKey with POD value", "[blob_store][typed]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    InMemoryBlobStore raw_store(fixture.allocator);
    BlobStore<InMemoryBlobStore> store(&raw_store);

    SECTION("Put and get with ChunkKey") {
        DatasetId dataset(1);
        std::array<uint64_t, 3> coords = {0, 0, 0};
        ChunkKey key(dataset, coords);
        uint64_t chunk_size = 4096;

        REQUIRE(store.PutBlob(key, chunk_size));

        auto result = store.GetBlob<ChunkKey, uint64_t>(key);
        REQUIRE(result.has_value());
        REQUIRE(*result == chunk_size);
    }

    SECTION("Different chunk coordinates") {
        DatasetId dataset(1);

        std::array<uint64_t, 2> coords1 = {0, 0};
        std::array<uint64_t, 2> coords2 = {0, 1};
        std::array<uint64_t, 2> coords3 = {1, 0};

        ChunkKey key1(dataset, coords1);
        ChunkKey key2(dataset, coords2);
        ChunkKey key3(dataset, coords3);

        store.PutBlob(key1, uint64_t{100});
        store.PutBlob(key2, uint64_t{200});
        store.PutBlob(key3, uint64_t{300});

        REQUIRE(raw_store.Size() == 3);

        auto r1 = store.GetBlob<ChunkKey, uint64_t>(key1);
        auto r2 = store.GetBlob<ChunkKey, uint64_t>(key2);
        auto r3 = store.GetBlob<ChunkKey, uint64_t>(key3);

        REQUIRE(*r1 == 100);
        REQUIRE(*r2 == 200);
        REQUIRE(*r3 == 300);
    }
}

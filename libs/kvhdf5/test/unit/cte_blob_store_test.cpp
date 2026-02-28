#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/cte_blob_store.h>
#include <kvhdf5/blob_store.h>
#include <kvhdf5/types.h>
#include <kvhdf5/group.h>
#include <cuda/std/algorithm>
#include "../common/cte_runtime.h"
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("CteBlobStore - Raw API", "[cte][blob_store]") {
    EnsureCteRuntime();
    CteBlobStore store("cte_raw_api_test");

    SECTION("Basic put and get") {
        std::array<byte_t, 4> key = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
        std::array<byte_t, 8> value = {
            byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40},
            byte_t{50}, byte_t{60}, byte_t{70}, byte_t{80}
        };

        REQUIRE(store.PutBlob(key, value));

        std::array<byte_t, 8> output;
        auto result = store.GetBlob(key, output);

        REQUIRE(result.has_value());
        REQUIRE(result->size() == 8);
        REQUIRE(cstd::equal(result->begin(), result->end(), value.begin()));
    }

    SECTION("Overwrite existing key with same size") {
        std::array<byte_t, 4> key = {byte_t{5}, byte_t{6}, byte_t{7}, byte_t{8}};
        std::array<byte_t, 4> value1 = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};
        std::array<byte_t, 4> value2 = {byte_t{99}, byte_t{88}, byte_t{77}, byte_t{66}};

        REQUIRE(store.PutBlob(key, value1));
        REQUIRE(store.PutBlob(key, value2));

        std::array<byte_t, 4> output;
        auto result = store.GetBlob(key, output);

        REQUIRE(result.has_value());
        REQUIRE(cstd::equal(result->begin(), result->end(), value2.begin()));
    }

    SECTION("Overwrite with smaller value") {
        // CTE doesn't truncate blobs on overwrite at offset 0,
        // so CteBlobStore must delete-then-put to handle this correctly.
        std::array<byte_t, 4> key = {byte_t{9}, byte_t{10}, byte_t{11}, byte_t{12}};
        std::array<byte_t, 8> large_value = {
            byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4},
            byte_t{5}, byte_t{6}, byte_t{7}, byte_t{8}
        };
        std::array<byte_t, 4> small_value = {byte_t{99}, byte_t{88}, byte_t{77}, byte_t{66}};

        REQUIRE(store.PutBlob(key, large_value));
        REQUIRE(store.PutBlob(key, small_value));

        std::array<byte_t, 4> output;
        auto result = store.GetBlob(key, output);

        REQUIRE(result.has_value());
        REQUIRE(result->size() == 4);
        REQUIRE(cstd::equal(result->begin(), result->end(), small_value.begin()));
    }

    SECTION("Get with insufficient buffer") {
        std::array<byte_t, 4> key = {byte_t{13}, byte_t{14}, byte_t{15}, byte_t{16}};
        std::array<byte_t, 8> value = {
            byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40},
            byte_t{50}, byte_t{60}, byte_t{70}, byte_t{80}
        };

        REQUIRE(store.PutBlob(key, value));

        std::array<byte_t, 4> small_output;
        auto result = store.GetBlob(key, small_output);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotEnoughSpace);
    }

    SECTION("Get non-existent key") {
        std::array<byte_t, 4> key = {byte_t{0xFF}, byte_t{0xFE}, byte_t{0xFD}, byte_t{0xFC}};
        std::array<byte_t, 8> output;

        auto result = store.GetBlob(key, output);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Delete blob") {
        std::array<byte_t, 4> key = {byte_t{17}, byte_t{18}, byte_t{19}, byte_t{20}};
        std::array<byte_t, 4> value = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};

        REQUIRE(store.PutBlob(key, value));
        REQUIRE(store.Exists(key));

        REQUIRE(store.DeleteBlob(key));
        REQUIRE_FALSE(store.Exists(key));

        REQUIRE_FALSE(store.DeleteBlob(key));
    }

    SECTION("Exists check") {
        std::array<byte_t, 4> key1 = {byte_t{21}, byte_t{22}, byte_t{23}, byte_t{24}};
        std::array<byte_t, 4> key2 = {byte_t{25}, byte_t{26}, byte_t{27}, byte_t{28}};
        std::array<byte_t, 4> value = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};

        REQUIRE_FALSE(store.Exists(key1));

        store.PutBlob(key1, value);

        REQUIRE(store.Exists(key1));
        REQUIRE_FALSE(store.Exists(key2));
    }

    SECTION("Multiple entries") {
        std::array<byte_t, 4> key1 = {byte_t{30}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> key2 = {byte_t{31}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> key3 = {byte_t{32}, byte_t{0}, byte_t{0}, byte_t{0}};

        std::array<byte_t, 4> val1 = {byte_t{1}, byte_t{1}, byte_t{1}, byte_t{1}};
        std::array<byte_t, 4> val2 = {byte_t{2}, byte_t{2}, byte_t{2}, byte_t{2}};
        std::array<byte_t, 4> val3 = {byte_t{3}, byte_t{3}, byte_t{3}, byte_t{3}};

        store.PutBlob(key1, val1);
        store.PutBlob(key2, val2);
        store.PutBlob(key3, val3);

        REQUIRE(store.Exists(key1));
        REQUIRE(store.Exists(key2));
        REQUIRE(store.Exists(key3));

        std::array<byte_t, 4> out;

        auto r1 = store.GetBlob(key1, out);
        REQUIRE(r1.has_value());
        REQUIRE(cstd::equal(r1->begin(), r1->end(), val1.begin()));

        auto r2 = store.GetBlob(key2, out);
        REQUIRE(r2.has_value());
        REQUIRE(cstd::equal(r2->begin(), r2->end(), val2.begin()));

        auto r3 = store.GetBlob(key3, out);
        REQUIRE(r3.has_value());
        REQUIRE(cstd::equal(r3->begin(), r3->end(), val3.begin()));
    }

    SECTION("Single byte key and value") {
        std::array<byte_t, 1> key = {byte_t{0xAA}};
        std::array<byte_t, 1> value = {byte_t{0xBB}};

        REQUIRE(store.PutBlob(key, value));

        std::array<byte_t, 1> output;
        auto result = store.GetBlob(key, output);

        REQUIRE(result.has_value());
        REQUIRE(result->size() == 1);
        REQUIRE((*result)[0] == byte_t{0xBB});
    }

    SECTION("GetBlob after delete returns NotExist") {
        std::array<byte_t, 4> key = {byte_t{40}, byte_t{41}, byte_t{42}, byte_t{43}};
        std::array<byte_t, 4> value = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};

        REQUIRE(store.PutBlob(key, value));
        REQUIRE(store.DeleteBlob(key));

        std::array<byte_t, 4> output;
        auto result = store.GetBlob(key, output);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Put-delete-reput lifecycle") {
        std::array<byte_t, 4> key = {byte_t{44}, byte_t{45}, byte_t{46}, byte_t{47}};
        std::array<byte_t, 4> value1 = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};
        std::array<byte_t, 4> value2 = {byte_t{50}, byte_t{60}, byte_t{70}, byte_t{80}};

        REQUIRE(store.PutBlob(key, value1));
        REQUIRE(store.Exists(key));

        REQUIRE(store.DeleteBlob(key));
        REQUIRE_FALSE(store.Exists(key));

        REQUIRE(store.PutBlob(key, value2));
        REQUIRE(store.Exists(key));

        std::array<byte_t, 4> output;
        auto result = store.GetBlob(key, output);
        REQUIRE(result.has_value());
        REQUIRE(cstd::equal(result->begin(), result->end(), value2.begin()));
    }

    SECTION("Buffer exactly the right size succeeds") {
        std::array<byte_t, 4> key = {byte_t{48}, byte_t{49}, byte_t{50}, byte_t{51}};
        std::array<byte_t, 6> value = {
            byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}, byte_t{5}, byte_t{6}
        };

        REQUIRE(store.PutBlob(key, value));

        std::array<byte_t, 6> exact_output;
        auto result = store.GetBlob(key, exact_output);
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 6);
        REQUIRE(cstd::equal(result->begin(), result->end(), value.begin()));

        std::array<byte_t, 5> short_output;
        auto fail_result = store.GetBlob(key, short_output);
        REQUIRE_FALSE(fail_result.has_value());
        REQUIRE(fail_result.error() == BlobStoreError::NotEnoughSpace);
    }

    SECTION("Keys differing by one byte are distinct") {
        std::array<byte_t, 4> key_a = {byte_t{0}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> key_b = {byte_t{0}, byte_t{0}, byte_t{0}, byte_t{1}};

        std::array<byte_t, 4> val_a = {byte_t{0xAA}, byte_t{0xAA}, byte_t{0xAA}, byte_t{0xAA}};
        std::array<byte_t, 4> val_b = {byte_t{0xBB}, byte_t{0xBB}, byte_t{0xBB}, byte_t{0xBB}};

        store.PutBlob(key_a, val_a);
        store.PutBlob(key_b, val_b);

        std::array<byte_t, 4> out;

        auto ra = store.GetBlob(key_a, out);
        REQUIRE(ra.has_value());
        REQUIRE(cstd::equal(ra->begin(), ra->end(), val_a.begin()));

        auto rb = store.GetBlob(key_b, out);
        REQUIRE(rb.has_value());
        REQUIRE(cstd::equal(rb->begin(), rb->end(), val_b.begin()));
    }

    SECTION("All-zero and all-FF keys") {
        std::array<byte_t, 4> zero_key = {byte_t{0}, byte_t{0}, byte_t{0}, byte_t{0}};
        std::array<byte_t, 4> ff_key   = {byte_t{0xFF}, byte_t{0xFF}, byte_t{0xFF}, byte_t{0xFF}};

        std::array<byte_t, 2> val_z = {byte_t{1}, byte_t{2}};
        std::array<byte_t, 2> val_f = {byte_t{3}, byte_t{4}};

        REQUIRE(store.PutBlob(zero_key, val_z));
        REQUIRE(store.PutBlob(ff_key, val_f));

        std::array<byte_t, 2> out;

        auto rz = store.GetBlob(zero_key, out);
        REQUIRE(rz.has_value());
        REQUIRE(cstd::equal(rz->begin(), rz->end(), val_z.begin()));

        auto rf = store.GetBlob(ff_key, out);
        REQUIRE(rf.has_value());
        REQUIRE(cstd::equal(rf->begin(), rf->end(), val_f.begin()));
    }
}

TEST_CASE("CteBlobStore - Large data", "[cte][blob_store]") {
    EnsureCteRuntime();
    CteBlobStore store("cte_raw_large_test");

    SECTION("Put and get large blob") {
        std::array<byte_t, 4> key = {byte_t{1}, byte_t{0}, byte_t{0}, byte_t{0}};

        // Create a ~4KB value
        constexpr size_t kLargeSize = 4096;
        std::array<byte_t, kLargeSize> value;
        for (size_t i = 0; i < kLargeSize; ++i) {
            value[i] = byte_t(i % 256);
        }

        REQUIRE(store.PutBlob(key, value));

        std::array<byte_t, kLargeSize> output;
        auto result = store.GetBlob(key, output);

        REQUIRE(result.has_value());
        REQUIRE(result->size() == kLargeSize);
        REQUIRE(cstd::equal(result->begin(), result->end(), value.begin()));
    }
}

TEST_CASE("CteBlobStore - Empty value", "[cte][blob_store]") {
    EnsureCteRuntime();
    CteBlobStore store("cte_edge_empty_value");

    // CTE's GetBlob requires data_size > 0, so the implementation
    // must handle zero-length values specially (e.g. sentinel byte).
    std::array<byte_t, 4> key = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
    cstd::span<const byte_t> empty_value;

    REQUIRE(store.PutBlob(key, empty_value));
    REQUIRE(store.Exists(key));

    std::array<byte_t, 0> output;
    auto result = store.GetBlob(key, output);
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 0);
}

TEST_CASE("CteBlobStore - Empty key", "[cte][blob_store]") {
    EnsureCteRuntime();
    CteBlobStore store("cte_edge_empty_key");

    // Empty key → empty hex string → empty CTE blob name.
    // Implementation may reject this or handle it; test documents behavior.
    cstd::span<const byte_t> empty_key;
    std::array<byte_t, 4> value = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};

    REQUIRE(store.PutBlob(empty_key, value));
    REQUIRE(store.Exists(empty_key));

    std::array<byte_t, 4> output;
    auto result = store.GetBlob(empty_key, output);
    REQUIRE(result.has_value());
    REQUIRE(cstd::equal(result->begin(), result->end(), value.begin()));
}

TEST_CASE("CteBlobStore - Two instances sharing a tag", "[cte][blob_store]") {
    EnsureCteRuntime();

    // Two CteBlobStore objects pointing at the same CTE tag
    // should see each other's data.
    CteBlobStore store_a("cte_edge_shared_tag");
    CteBlobStore store_b("cte_edge_shared_tag");

    std::array<byte_t, 4> key = {byte_t{1}, byte_t{2}, byte_t{3}, byte_t{4}};
    std::array<byte_t, 4> value = {byte_t{10}, byte_t{20}, byte_t{30}, byte_t{40}};

    REQUIRE(store_a.PutBlob(key, value));

    // store_b should be able to read what store_a wrote
    REQUIRE(store_b.Exists(key));

    std::array<byte_t, 4> output;
    auto result = store_b.GetBlob(key, output);
    REQUIRE(result.has_value());
    REQUIRE(cstd::equal(result->begin(), result->end(), value.begin()));
}

TEST_CASE("BlobStore<CteBlobStore> - POD types", "[cte][blob_store][typed]") {
    EnsureCteRuntime();
    CteBlobStore raw_store("cte_typed_pod_test");
    BlobStore<CteBlobStore> store(&raw_store);

    SECTION("Put and get POD key-value") {
        ObjectId key(42);
        uint64_t value = 12345;

        REQUIRE(store.PutBlob(key, value));

        auto result = store.GetBlob<ObjectId, uint64_t>(key);
        REQUIRE(result.has_value());
        REQUIRE(*result == value);
    }

    SECTION("Get non-existent POD key") {
        ObjectId key(9999);

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

TEST_CASE("BlobStore<CteBlobStore> - Custom value serialization", "[cte][blob_store][typed]") {
    EnsureCteRuntime();
    CteBlobStore raw_store("cte_typed_custom_test");
    BlobStore<CteBlobStore> store(&raw_store);

    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    Context ctx(fixture.allocator);

    SECTION("Put and get GroupMetadata") {
        GroupId key(1);
        GroupMetadata metadata{key, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};

        metadata.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(10)), "test_dataset"));

        bool put_result = store.PutBlob(
            key,
            metadata,
            [](auto& s, const auto& v) { v.Serialize(s); }
        );
        REQUIRE(put_result);

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
        GroupId key1(101);
        GroupId key2(102);

        GroupMetadata meta1{key1, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta1.children.push_back(GroupEntry::NewGroup(GroupId(ObjectId(100)), "child1"));

        GroupMetadata meta2{key2, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta2.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(200)), "child2"));

        store.PutBlob(key1, meta1, [](auto& s, const auto& v) { v.Serialize(s); });
        store.PutBlob(key2, meta2, [](auto& s, const auto& v) { v.Serialize(s); });

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

    SECTION("Overwrite GroupMetadata") {
        GroupId key(105);

        GroupMetadata meta1{key, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta1.children.push_back(GroupEntry::NewGroup(GroupId(ObjectId(10)), "old"));

        GroupMetadata meta2{key, vector<GroupEntry>(fixture.allocator), vector<Attribute>(fixture.allocator)};
        meta2.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(20)), "new"));

        store.PutBlob(key, meta1, [](auto& s, const auto& v) { v.Serialize(s); });
        store.PutBlob(key, meta2, [](auto& s, const auto& v) { v.Serialize(s); });

        auto result = store.GetBlob<GroupId, GroupMetadata>(
            key, [&ctx](auto& d) { return GroupMetadata::Deserialize(d, ctx); });

        REQUIRE(result.has_value());
        REQUIRE(result->children.size() == 1);
        REQUIRE(result->children[0].name == "new");
    }
}

TEST_CASE("BlobStore<CteBlobStore> - ChunkKey with POD value", "[cte][blob_store][typed]") {
    EnsureCteRuntime();
    CteBlobStore raw_store("cte_typed_chunk_test");
    BlobStore<CteBlobStore> store(&raw_store);

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

        auto r1 = store.GetBlob<ChunkKey, uint64_t>(key1);
        auto r2 = store.GetBlob<ChunkKey, uint64_t>(key2);
        auto r3 = store.GetBlob<ChunkKey, uint64_t>(key3);

        REQUIRE(*r1 == 100);
        REQUIRE(*r2 == 200);
        REQUIRE(*r3 == 300);
    }
}

static_assert(RawBlobStore<CteBlobStore>);

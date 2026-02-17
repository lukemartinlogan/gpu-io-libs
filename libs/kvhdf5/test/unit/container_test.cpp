#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/container.h>
#include <kvhdf5/memory_blob_store.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("Container - Initialization", "[container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);

    SECTION("Root group is created and valid") {
        GroupId root = container.RootGroup();
        REQUIRE(root.IsValid());

        // Root group should exist in storage
        REQUIRE(container.GroupExists(root));

        // Should be able to retrieve root group metadata
        auto result = container.GetGroup(root);
        REQUIRE(result.has_value());
        REQUIRE(result->id == root);
        REQUIRE(result->children.size() == 0);
        REQUIRE(result->attributes.size() == 0);
    }

    SECTION("Context is initialized") {
        REQUIRE(&container.GetContext().GetAllocator() == fixture.allocator);
    }
}

TEST_CASE("Container - ID Allocation", "[container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);

    SECTION("AllocateId returns sequential IDs") {
        ObjectId id1 = container.AllocateId();
        ObjectId id2 = container.AllocateId();
        ObjectId id3 = container.AllocateId();

        REQUIRE(id1.IsValid());
        REQUIRE(id2.IsValid());
        REQUIRE(id3.IsValid());

        // IDs should be different
        REQUIRE(id1 != id2);
        REQUIRE(id2 != id3);
        REQUIRE(id1 != id3);

        // IDs should be sequential (root group takes the first ID)
        REQUIRE(id2.id == id1.id + 1);
        REQUIRE(id3.id == id2.id + 1);
    }

    SECTION("First allocated ID is not 0") {
        ObjectId id = container.AllocateId();
        REQUIRE(id.id != 0);
    }
}

TEST_CASE("Container - Group Operations", "[container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);

    SECTION("Put and get group metadata") {
        GroupId group_id = GroupId(container.AllocateId());

        GroupMetadata metadata{
            group_id,
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        };

        // Put group
        REQUIRE(container.PutGroup(group_id, metadata));

        // Group should exist
        REQUIRE(container.GroupExists(group_id));

        // Get group
        auto result = container.GetGroup(group_id);
        REQUIRE(result.has_value());
        REQUIRE(result->id == group_id);
    }

    SECTION("Get non-existent group returns error") {
        GroupId nonexistent_id = GroupId(ObjectId(9999));

        auto result = container.GetGroup(nonexistent_id);
        REQUIRE(!result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Delete group") {
        GroupId group_id = GroupId(container.AllocateId());

        GroupMetadata metadata{
            group_id,
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        };

        REQUIRE(container.PutGroup(group_id, metadata));
        REQUIRE(container.GroupExists(group_id));

        // Delete
        REQUIRE(container.DeleteGroup(group_id));
        REQUIRE(!container.GroupExists(group_id));

        // Get should fail
        auto result = container.GetGroup(group_id);
        REQUIRE(!result.has_value());
    }

    SECTION("Update group metadata (overwrite)") {
        GroupId group_id = GroupId(container.AllocateId());

        // Create initial metadata
        GroupMetadata metadata1{
            group_id,
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        };

        REQUIRE(container.PutGroup(group_id, metadata1));

        // Create updated metadata with a child
        GroupMetadata metadata2{
            group_id,
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        };

        GroupEntry child_entry;
        child_entry.kind = ChildKind::Group;
        child_entry.object_id = ObjectId(container.AllocateId());
        child_entry.name = gpu_string<255>("child_group");
        metadata2.children.push_back(child_entry);

        // Overwrite
        REQUIRE(container.PutGroup(group_id, metadata2));

        // Verify updated metadata
        auto result = container.GetGroup(group_id);
        REQUIRE(result.has_value());
        REQUIRE(result->children.size() == 1);
        REQUIRE(result->children[0].name == "child_group");
    }
}

TEST_CASE("Container - Dataset Operations", "[container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);

    SECTION("Put and get dataset metadata") {
        DatasetId dataset_id = DatasetId(container.AllocateId());

        auto shape_result = DatasetShape::Create({100, 200}, {10, 20});
        REQUIRE(shape_result.has_value());

        DatasetMetadata metadata{
            dataset_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float32)),
            *shape_result,
            vector<Attribute>(fixture.allocator)
        };

        // Put dataset
        REQUIRE(container.PutDataset(dataset_id, metadata));

        // Dataset should exist
        REQUIRE(container.DatasetExists(dataset_id));

        // Get dataset
        auto result = container.GetDataset(dataset_id);
        REQUIRE(result.has_value());
        REQUIRE(result->id == dataset_id);
        REQUIRE(result->datatype == PrimitiveType::Kind::Float32);
        REQUIRE(result->shape == *shape_result);
    }

    SECTION("Get non-existent dataset returns error") {
        DatasetId nonexistent_id = DatasetId(ObjectId(9999));

        auto result = container.GetDataset(nonexistent_id);
        REQUIRE(!result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Delete dataset") {
        DatasetId dataset_id = DatasetId(container.AllocateId());

        auto shape_result = DatasetShape::Create({100}, {10});
        REQUIRE(shape_result.has_value());

        DatasetMetadata metadata{
            dataset_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            *shape_result,
            vector<Attribute>(fixture.allocator)
        };

        REQUIRE(container.PutDataset(dataset_id, metadata));
        REQUIRE(container.DatasetExists(dataset_id));

        // Delete
        REQUIRE(container.DeleteDataset(dataset_id));
        REQUIRE(!container.DatasetExists(dataset_id));

        // Get should fail
        auto result = container.GetDataset(dataset_id);
        REQUIRE(!result.has_value());
    }
}

TEST_CASE("Container - Datatype Operations", "[container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);

    SECTION("Put and get complex datatype descriptor") {
        DatatypeId datatype_id = DatatypeId(container.AllocateId());

        ComplexDatatypeDescriptor descriptor;
        descriptor.kind = ComplexDatatypeDescriptor::Kind::Compound;
        descriptor.element_size = 32;

        // Put datatype
        REQUIRE(container.PutDatatype(datatype_id, descriptor));

        // Datatype should exist
        REQUIRE(container.DatatypeExists(datatype_id));

        // Get datatype
        auto result = container.GetDatatype(datatype_id);
        REQUIRE(result.has_value());
        REQUIRE(result->kind == ComplexDatatypeDescriptor::Kind::Compound);
        REQUIRE(result->element_size == 32);
    }

    SECTION("Get non-existent datatype returns error") {
        DatatypeId nonexistent_id = DatatypeId(ObjectId(9999));

        auto result = container.GetDatatype(nonexistent_id);
        REQUIRE(!result.has_value());
        REQUIRE(result.error() == BlobStoreError::NotExist);
    }

    SECTION("Delete datatype") {
        DatatypeId datatype_id = DatatypeId(container.AllocateId());

        ComplexDatatypeDescriptor descriptor;
        descriptor.kind = ComplexDatatypeDescriptor::Kind::Array;
        descriptor.element_size = 16;

        REQUIRE(container.PutDatatype(datatype_id, descriptor));
        REQUIRE(container.DatatypeExists(datatype_id));

        // Delete
        REQUIRE(container.DeleteDatatype(datatype_id));
        REQUIRE(!container.DatatypeExists(datatype_id));

        // Get should fail
        auto result = container.GetDatatype(datatype_id);
        REQUIRE(!result.has_value());
    }
}

TEST_CASE("Container - Multiple Containers", "[container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container1(fixture.allocator);
    Container<InMemoryBlobStore> container2(fixture.allocator);

    SECTION("Containers have independent storage") {
        // Use the same ID in both containers to prove storage is separate
        GroupId shared_id = GroupId(ObjectId(999));

        GroupMetadata metadata1{
            shared_id,
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        };

        GroupMetadata metadata2{
            shared_id,
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        };

        // Add a child to metadata2 to differentiate it
        GroupEntry child_entry;
        child_entry.kind = ChildKind::Group;
        child_entry.object_id = ObjectId(1000);
        child_entry.name = gpu_string<255>("child");
        metadata2.children.push_back(child_entry);

        REQUIRE(container1.PutGroup(shared_id, metadata1));
        REQUIRE(container2.PutGroup(shared_id, metadata2));

        // Both containers can store the same ID, but the data is independent
        auto result1 = container1.GetGroup(shared_id);
        auto result2 = container2.GetGroup(shared_id);

        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        // Container1 has no children, container2 has one child
        REQUIRE(result1->children.size() == 0);
        REQUIRE(result2->children.size() == 1);
    }

    SECTION("Containers have independent ID allocation") {
        ObjectId id1 = container1.AllocateId();
        ObjectId id2 = container2.AllocateId();

        // IDs might coincide since containers are independent
        // But they should both be valid
        REQUIRE(id1.IsValid());
        REQUIRE(id2.IsValid());
    }
}

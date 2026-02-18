#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/container.h>
#include <kvhdf5/memory_blob_store.h>
#include <utils/algorithms.h>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

static_assert(ProvidesAllocator<Container<InMemoryBlobStore>>);

// helper to use strings with POD ctor for Attribute
template<size_t N>
CROSS_FUN constexpr cstd::array<char, N> as_array(const char (&str)[N]) {
    cstd::array<char, N> result{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = str[i];
    }
    return result;
}

TEST_CASE("Integration - Complex Nested Hierarchy", "[integration][container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);

    /*
     * Build this hierarchy:
     *
     * / (root)
     *   ├─ experiments/ (group)
     *   │   ├─ exp001/ (group) [attrs: date="2024-01-15", researcher="Alice"]
     *   │   │   ├─ data (dataset: float32, shape=[1000,3], chunks=[100,3])
     *   │   │   │   [attrs: units="meters", description="Position data"]
     *   │   │   └─ labels (dataset: int32, shape=[1000], chunks=[100])
     *   │   │       [attrs: description="Class labels"]
     *   │   └─ exp002/ (group) [attrs: date="2024-01-20", researcher="Bob"]
     *   │       ├─ raw_data (dataset: float64, shape=[500,500,10], chunks=[50,50,10])
     *   │       │   [attrs: units="volts", sampling_rate=1000.0]
     *   │       ├─ processed_data (dataset: float32, shape=[500,500,10], chunks=[50,50,10])
     *   │       │   [attrs: method="FFT", normalized=true]
     *   │       └─ metadata/ (group)
     *   │           └─ calibration (dataset: float64, shape=[10], chunks=[10])
     *   │               [attrs: version="1.0", device_id=42]
     *   └─ reference/ (group) [attrs: purpose="baseline"]
     *       └─ standards (dataset: float32, shape=[100], chunks=[10])
     *           [attrs: standard="ISO-9001", verified=true]
     */

    GroupId root = container.RootGroup();

    // ===== Create /experiments group =====
    GroupId experiments_id = GroupId(container.AllocateId());
    {
        GroupMetadata experiments{
            experiments_id,
            vector<GroupEntry>(&container.GetAllocator()),
            vector<Attribute>(&container.GetAllocator())
        };
        REQUIRE(container.PutGroup(experiments_id, experiments));

        // Add experiments to root
        auto root_meta = container.GetGroup(root);
        REQUIRE(root_meta.has_value());

        GroupEntry experiments_entry = GroupEntry::NewGroup(experiments_id, "experiments");
        root_meta->children.push_back(experiments_entry);

        REQUIRE(container.PutGroup(root, *root_meta));
    }

    // ===== Create /experiments/exp001 group =====
    GroupId exp001_id = GroupId(container.AllocateId());
    {
        GroupMetadata exp001{
            exp001_id,
            vector<GroupEntry>(&container.GetAllocator()),
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes to exp001
        int32_t date_value = 20240115;
        Attribute date_attr(
            "date",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            date_value
        );
        exp001.attributes.push_back(date_attr);

        Attribute researcher_attr(
            "researcher",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("Alice")
        );
        exp001.attributes.push_back(researcher_attr);

        REQUIRE(container.PutGroup(exp001_id, exp001));

        // Add exp001 to experiments
        auto experiments_meta = container.GetGroup(experiments_id);
        REQUIRE(experiments_meta.has_value());

        GroupEntry exp001_entry = GroupEntry::NewGroup(exp001_id, "exp001");
        experiments_meta->children.push_back(exp001_entry);

        REQUIRE(container.PutGroup(experiments_id, *experiments_meta));
    }

    // ===== Create /experiments/exp001/data dataset =====
    DatasetId exp001_data_id = DatasetId(container.AllocateId());
    {
        auto shape_result = DatasetShape::Create({1000, 3}, {100, 3});
        REQUIRE(shape_result.has_value());

        DatasetMetadata data_meta{
            exp001_data_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float32)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes
        Attribute units_attr(
            "units",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("meters")
        );
        data_meta.attributes.push_back(units_attr);

        Attribute desc_attr(
            "description",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("Position data")
        );
        data_meta.attributes.push_back(desc_attr);

        REQUIRE(container.PutDataset(exp001_data_id, data_meta));

        // Add to exp001
        auto exp001_meta = container.GetGroup(exp001_id);
        REQUIRE(exp001_meta.has_value());

        GroupEntry data_entry = GroupEntry::NewDataset(exp001_data_id, "data");
        exp001_meta->children.push_back(data_entry);

        REQUIRE(container.PutGroup(exp001_id, *exp001_meta));
    }

    // ===== Create /experiments/exp001/labels dataset =====
    DatasetId exp001_labels_id = DatasetId(container.AllocateId());
    {
        auto shape_result = DatasetShape::Create({1000}, {100});
        REQUIRE(shape_result.has_value());

        DatasetMetadata labels_meta{
            exp001_labels_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };

        // Add description attribute
        Attribute desc_attr(
            "description",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("Class labels")
        );
        labels_meta.attributes.push_back(desc_attr);

        REQUIRE(container.PutDataset(exp001_labels_id, labels_meta));

        // Add to exp001
        auto exp001_meta = container.GetGroup(exp001_id);
        REQUIRE(exp001_meta.has_value());

        GroupEntry labels_entry = GroupEntry::NewDataset(exp001_labels_id, "labels");
        exp001_meta->children.push_back(labels_entry);

        REQUIRE(container.PutGroup(exp001_id, *exp001_meta));
    }

    // ===== Create /experiments/exp002 group =====
    GroupId exp002_id = GroupId(container.AllocateId());
    {
        GroupMetadata exp002{
            exp002_id,
            vector<GroupEntry>(&container.GetAllocator()),
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes
        int32_t date_value = 20240120;
        Attribute date_attr(
            "date",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            date_value
        );
        exp002.attributes.push_back(date_attr);

        Attribute researcher_attr(
            "researcher",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("Bob")
        );
        exp002.attributes.push_back(researcher_attr);

        REQUIRE(container.PutGroup(exp002_id, exp002));

        // Add exp002 to experiments
        auto experiments_meta = container.GetGroup(experiments_id);
        REQUIRE(experiments_meta.has_value());

        GroupEntry exp002_entry = GroupEntry::NewGroup(exp002_id, "exp002");
        experiments_meta->children.push_back(exp002_entry);

        REQUIRE(container.PutGroup(experiments_id, *experiments_meta));
    }

    // ===== Create /experiments/exp002/raw_data dataset =====
    DatasetId exp002_raw_id = DatasetId(container.AllocateId());
    {
        auto shape_result = DatasetShape::Create({500, 500, 10}, {50, 50, 10});
        REQUIRE(shape_result.has_value());

        DatasetMetadata raw_meta{
            exp002_raw_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float64)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes
        Attribute units_attr(
            "units",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("volts")
        );
        raw_meta.attributes.push_back(units_attr);

        double sampling_rate = 1000.0;
        Attribute sampling_attr(
            "sampling_rate",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float64)),
            sampling_rate
        );
        raw_meta.attributes.push_back(sampling_attr);

        REQUIRE(container.PutDataset(exp002_raw_id, raw_meta));

        // Add to exp002
        auto exp002_meta = container.GetGroup(exp002_id);
        REQUIRE(exp002_meta.has_value());

        GroupEntry raw_entry = GroupEntry::NewDataset(exp002_raw_id, "raw_data");
        exp002_meta->children.push_back(raw_entry);

        REQUIRE(container.PutGroup(exp002_id, *exp002_meta));
    }

    // ===== Create /experiments/exp002/processed_data dataset =====
    DatasetId exp002_processed_id = DatasetId(container.AllocateId());
    {
        auto shape_result = DatasetShape::Create({500, 500, 10}, {50, 50, 10});
        REQUIRE(shape_result.has_value());

        DatasetMetadata processed_meta{
            exp002_processed_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float32)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes
        Attribute method_attr(
            "method",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("FFT")
        );
        processed_meta.attributes.push_back(method_attr);

        uint8_t normalized_value = 1;  // true
        Attribute normalized_attr("normalized",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Uint8)),
            normalized_value
        );
        processed_meta.attributes.push_back(normalized_attr);

        REQUIRE(container.PutDataset(exp002_processed_id, processed_meta));

        // Add to exp002
        auto exp002_meta = container.GetGroup(exp002_id);
        REQUIRE(exp002_meta.has_value());

        GroupEntry processed_entry = GroupEntry::NewDataset(exp002_processed_id, "processed_data");
        exp002_meta->children.push_back(processed_entry);

        REQUIRE(container.PutGroup(exp002_id, *exp002_meta));
    }

    // ===== Create /experiments/exp002/metadata group =====
    GroupId exp002_metadata_id = GroupId(container.AllocateId());
    {
        GroupMetadata metadata_group{
            exp002_metadata_id,
            vector<GroupEntry>(&container.GetAllocator()),
            vector<Attribute>(&container.GetAllocator())
        };

        REQUIRE(container.PutGroup(exp002_metadata_id, metadata_group));

        // Add to exp002
        auto exp002_meta = container.GetGroup(exp002_id);
        REQUIRE(exp002_meta.has_value());

        GroupEntry metadata_entry = GroupEntry::NewGroup(exp002_metadata_id, "metadata");
        exp002_meta->children.push_back(metadata_entry);

        REQUIRE(container.PutGroup(exp002_id, *exp002_meta));
    }

    // ===== Create /experiments/exp002/metadata/calibration dataset =====
    DatasetId calibration_id = DatasetId(container.AllocateId());
    {
        auto shape_result = DatasetShape::Create({10}, {10});
        REQUIRE(shape_result.has_value());

        DatasetMetadata calibration_meta{
            calibration_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float64)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes
        Attribute version_attr(
            "version",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("1.0")
        );
        calibration_meta.attributes.push_back(version_attr);

        int32_t device_id = 42;
        Attribute device_attr("device_id",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            device_id
        );
        calibration_meta.attributes.push_back(device_attr);

        REQUIRE(container.PutDataset(calibration_id, calibration_meta));

        // Add to metadata group
        auto metadata_meta = container.GetGroup(exp002_metadata_id);
        REQUIRE(metadata_meta.has_value());

        GroupEntry calibration_entry = GroupEntry::NewDataset(calibration_id, "calibration");
        metadata_meta->children.push_back(calibration_entry);

        REQUIRE(container.PutGroup(exp002_metadata_id, *metadata_meta));
    }

    // ===== Create /reference group =====
    GroupId reference_id = GroupId(container.AllocateId());
    {
        GroupMetadata reference{
            reference_id,
            vector<GroupEntry>(&container.GetAllocator()),
            vector<Attribute>(&container.GetAllocator())
        };

        // Add purpose attribute
        Attribute purpose_attr(
            "purpose",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("baseline")
        );
        reference.attributes.push_back(purpose_attr);

        REQUIRE(container.PutGroup(reference_id, reference));

        // Add reference to root
        auto root_meta = container.GetGroup(root);
        REQUIRE(root_meta.has_value());

        GroupEntry reference_entry = GroupEntry::NewGroup(reference_id, "reference");
        root_meta->children.push_back(reference_entry);

        REQUIRE(container.PutGroup(root, *root_meta));
    }

    // ===== Create /reference/standards dataset =====
    DatasetId standards_id = DatasetId(container.AllocateId());
    {
        auto shape_result = DatasetShape::Create({100}, {10});
        REQUIRE(shape_result.has_value());

        DatasetMetadata standards_meta{
            standards_id,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float32)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };

        // Add attributes
        Attribute standard_attr(
            "standard",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8)),
            as_array("ISO-9001")
        );
        standards_meta.attributes.push_back(standard_attr);

        uint8_t verified_value = 1;  // true
        Attribute verified_attr(
            "verified",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Uint8)),
            verified_value
        );
        standards_meta.attributes.push_back(verified_attr);

        REQUIRE(container.PutDataset(standards_id, standards_meta));

        // Add to reference group
        auto reference_meta = container.GetGroup(reference_id);
        REQUIRE(reference_meta.has_value());

        GroupEntry standards_entry = GroupEntry::NewDataset(standards_id, "standards");
        reference_meta->children.push_back(standards_entry);

        REQUIRE(container.PutGroup(reference_id, *reference_meta));
    }

    // ===== VERIFICATION TESTS =====

    SECTION("Verify root structure") {
        auto root_meta = container.GetGroup(root);
        REQUIRE(root_meta.has_value());
        REQUIRE(root_meta->children.size() == 2);  // experiments and reference

        // Find experiments and reference
        bool found_experiments = false;
        bool found_reference = false;
        for (const auto& child : root_meta->children) {
            if (child.name == "experiments") {
                found_experiments = true;
                REQUIRE(child.kind == ChildKind::Group);
                REQUIRE(child.object_id == experiments_id.id);
            } else if (child.name == "reference") {
                found_reference = true;
                REQUIRE(child.kind == ChildKind::Group);
                REQUIRE(child.object_id == reference_id.id);
            }
        }
        REQUIRE(found_experiments);
        REQUIRE(found_reference);
    }

    SECTION("Verify /experiments structure") {
        auto experiments_meta = container.GetGroup(experiments_id);
        REQUIRE(experiments_meta.has_value());
        REQUIRE(experiments_meta->children.size() == 2);  // exp001 and exp002
    }

    SECTION("Verify /experiments/exp001 attributes") {
        auto exp001_meta = container.GetGroup(exp001_id);
        REQUIRE(exp001_meta.has_value());
        REQUIRE(exp001_meta->attributes.size() == 2);

        // Check date attribute
        auto& date_attr = exp001_meta->attributes[0];
        REQUIRE(date_attr.name == "date");
        REQUIRE(date_attr.datatype == PrimitiveType::Kind::Int32);
        REQUIRE(date_attr.value.size() == 4);
        int32_t date_value;
        cstd::memcpy(&date_value, date_attr.value.data(), 4);
        REQUIRE(date_value == 20240115);

        // Check researcher attribute
        auto& researcher_attr = exp001_meta->attributes[1];
        REQUIRE(researcher_attr.name == "researcher");
        REQUIRE(researcher_attr.value.size() == 6);
        char researcher_name[7] = {0};
        cstd::memcpy(researcher_name, researcher_attr.value.data(), 6);
        REQUIRE(std::string(researcher_name) == "Alice");
    }

    SECTION("Verify /experiments/exp001 children") {
        auto exp001_meta = container.GetGroup(exp001_id);
        REQUIRE(exp001_meta.has_value());
        REQUIRE(exp001_meta->children.size() == 2);  // data and labels

        // Both should be datasets
        for (const auto& child : exp001_meta->children) {
            REQUIRE(child.kind == ChildKind::Dataset);
        }
    }

    SECTION("Verify /experiments/exp001/data dataset") {
        auto data_meta = container.GetDataset(exp001_data_id);
        REQUIRE(data_meta.has_value());
        REQUIRE(data_meta->datatype == PrimitiveType::Kind::Float32);
        REQUIRE(data_meta->shape.Ndims() == 2);
        REQUIRE(data_meta->shape.dims[0] == 1000);
        REQUIRE(data_meta->shape.dims[1] == 3);
        REQUIRE(data_meta->shape.chunk_dims[0] == 100);
        REQUIRE(data_meta->shape.chunk_dims[1] == 3);
        REQUIRE(data_meta->attributes.size() == 2);  // units and description
    }

    SECTION("Verify /experiments/exp002/metadata/calibration deep nesting") {
        // This tests 4-level deep nesting: root -> experiments -> exp002 -> metadata -> calibration
        auto calibration_meta = container.GetDataset(calibration_id);
        REQUIRE(calibration_meta.has_value());
        REQUIRE(calibration_meta->datatype == PrimitiveType::Kind::Float64);
        REQUIRE(calibration_meta->attributes.size() == 2);

        // Verify version attribute
        auto& version_attr = calibration_meta->attributes[0];
        REQUIRE(version_attr.name == "version");
        char version_str[5] = {0};
        cstd::memcpy(version_str, version_attr.value.data(), 4);
        REQUIRE(std::string(version_str) == "1.0");

        // Verify device_id attribute
        auto& device_attr = calibration_meta->attributes[1];
        REQUIRE(device_attr.name == "device_id");
        int32_t device_id;
        cstd::memcpy(&device_id, device_attr.value.data(), 4);
        REQUIRE(device_id == 42);
    }

    SECTION("Verify /reference/standards dataset attributes") {
        auto standards_meta = container.GetDataset(standards_id);
        REQUIRE(standards_meta.has_value());
        REQUIRE(standards_meta->attributes.size() == 2);

        // Check standard attribute
        auto& standard_attr = standards_meta->attributes[0];
        REQUIRE(standard_attr.name == "standard");
        char standard_str[10] = {0};
        cstd::memcpy(standard_str, standard_attr.value.data(), 9);
        REQUIRE(std::string(standard_str) == "ISO-9001");

        // Check verified attribute
        auto& verified_attr = standards_meta->attributes[1];
        REQUIRE(verified_attr.name == "verified");
        REQUIRE(verified_attr.value[0] == byte_t{1});
    }

    SECTION("Verify multi-dimensional dataset shapes") {
        // Test 3D dataset
        auto raw_data_meta = container.GetDataset(exp002_raw_id);
        REQUIRE(raw_data_meta.has_value());
        REQUIRE(raw_data_meta->shape.Ndims() == 3);
        REQUIRE(raw_data_meta->shape.dims[0] == 500);
        REQUIRE(raw_data_meta->shape.dims[1] == 500);
        REQUIRE(raw_data_meta->shape.dims[2] == 10);
        REQUIRE(raw_data_meta->shape.chunk_dims[0] == 50);
        REQUIRE(raw_data_meta->shape.chunk_dims[1] == 50);
        REQUIRE(raw_data_meta->shape.chunk_dims[2] == 10);
    }

    SECTION("Verify all objects are accessible") {
        // All objects created should be retrievable
        REQUIRE(container.GroupExists(root));
        REQUIRE(container.GroupExists(experiments_id));
        REQUIRE(container.GroupExists(exp001_id));
        REQUIRE(container.GroupExists(exp002_id));
        REQUIRE(container.GroupExists(exp002_metadata_id));
        REQUIRE(container.GroupExists(reference_id));

        REQUIRE(container.DatasetExists(exp001_data_id));
        REQUIRE(container.DatasetExists(exp001_labels_id));
        REQUIRE(container.DatasetExists(exp002_raw_id));
        REQUIRE(container.DatasetExists(exp002_processed_id));
        REQUIRE(container.DatasetExists(calibration_id));
        REQUIRE(container.DatasetExists(standards_id));
    }
}

TEST_CASE("Integration - Hierarchy Modification", "[integration][container]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    Container<InMemoryBlobStore> container(fixture.allocator);
    GroupId root = container.RootGroup();

    // Create initial structure
    GroupId group_a = GroupId(container.AllocateId());
    GroupMetadata group_a_meta{
        group_a,
        vector<GroupEntry>(&container.GetAllocator()),
        vector<Attribute>(&container.GetAllocator())
    };
    REQUIRE(container.PutGroup(group_a, group_a_meta));

    // Add to root
    auto root_meta = container.GetGroup(root);
    REQUIRE(root_meta.has_value());
    GroupEntry entry = GroupEntry::NewGroup(group_a, "group_a");
    root_meta->children.push_back(entry);
    REQUIRE(container.PutGroup(root, *root_meta));

    SECTION("Can add children to existing group") {
        DatasetId dataset = DatasetId(container.AllocateId());
        auto shape_result = DatasetShape::Create({100}, {10});
        REQUIRE(shape_result.has_value());

        DatasetMetadata dataset_meta{
            dataset,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float32)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };
        REQUIRE(container.PutDataset(dataset, dataset_meta));

        // Add dataset to group_a
        auto group_a_updated = container.GetGroup(group_a);
        REQUIRE(group_a_updated.has_value());

        GroupEntry dataset_entry = GroupEntry::NewDataset(dataset, "new_dataset");
        group_a_updated->children.push_back(dataset_entry);

        REQUIRE(container.PutGroup(group_a, *group_a_updated));

        // Verify
        auto final_meta = container.GetGroup(group_a);
        REQUIRE(final_meta.has_value());
        REQUIRE(final_meta->children.size() == 1);
        REQUIRE(final_meta->children[0].name == "new_dataset");
    }

    SECTION("Can remove children from group") {
        // Add a dataset first
        DatasetId dataset = DatasetId(container.AllocateId());
        auto shape_result = DatasetShape::Create({50}, {5});
        REQUIRE(shape_result.has_value());

        DatasetMetadata dataset_meta{
            dataset,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            *shape_result,
            vector<Attribute>(&container.GetAllocator())
        };
        REQUIRE(container.PutDataset(dataset, dataset_meta));

        auto group_a_meta = container.GetGroup(group_a);
        REQUIRE(group_a_meta.has_value());

        GroupEntry dataset_entry = GroupEntry::NewDataset(dataset, "temp_dataset");
        group_a_meta->children.push_back(dataset_entry);
        REQUIRE(container.PutGroup(group_a, *group_a_meta));

        // Now remove it
        auto updated_meta = container.GetGroup(group_a);
        REQUIRE(updated_meta.has_value());
        REQUIRE(updated_meta->children.size() == 1);

        updated_meta->children.clear();
        REQUIRE(container.PutGroup(group_a, *updated_meta));

        // Verify removal
        auto final_meta = container.GetGroup(group_a);
        REQUIRE(final_meta.has_value());
        REQUIRE(final_meta->children.size() == 0);
    }

    SECTION("Can modify attributes on existing objects") {
        // Add an attribute to group_a
        auto group_a_meta = container.GetGroup(group_a);
        REQUIRE(group_a_meta.has_value());

        int32_t version = 1;
        Attribute attr(
            "version",
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32)),
            version
        );
        group_a_meta->attributes.push_back(attr);
        REQUIRE(container.PutGroup(group_a, *group_a_meta));

        // Modify the attribute
        auto updated_meta = container.GetGroup(group_a);
        REQUIRE(updated_meta.has_value());
        REQUIRE(updated_meta->attributes.size() == 1);

        version = 2;
        cstd::memcpy(updated_meta->attributes[0].value.data(), &version, sizeof(version));
        REQUIRE(container.PutGroup(group_a, *updated_meta));

        // Verify modification
        auto final_meta = container.GetGroup(group_a);
        REQUIRE(final_meta.has_value());
        REQUIRE(final_meta->attributes.size() == 1);

        int32_t final_version;
        cstd::memcpy(&final_version, final_meta->attributes[0].value.data(), 4);
        REQUIRE(final_version == 2);
    }
}

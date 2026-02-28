#include <catch2/catch_test_macros.hpp>
#include "../../common/cte_runtime.h"
#include <cstring>
#include <vector>
#include <string>

TEST_CASE("CTE Tag creation and basic PutBlob/GetBlob", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    SECTION("Can create a Tag and write/read simple data") {
        wrp_cte::core::Tag tag("test_tag_basic");
        const char* blob_name = "simple_blob";
        const char* data = "Hello, CTE!";
        size_t data_size = strlen(data) + 1;

        // Write blob
        tag.PutBlob(blob_name, data, data_size);

        // Read blob
        char buffer[100] = {0};
        tag.GetBlob(blob_name, buffer, data_size);

        REQUIRE(strcmp(buffer, data) == 0);
    }
}

TEST_CASE("CTE binary data operations", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_binary");

    SECTION("Can write and read binary data") {
        const char* blob_name = "binary_blob";
        uint64_t data[] = {0x1234567890ABCDEF, 0xFEDCBA0987654321, 0xAAAAAAAAAAAAAAAA};
        size_t data_size = sizeof(data);

        // Write blob
        tag.PutBlob(blob_name, reinterpret_cast<const char*>(data), data_size);

        // Read blob
        uint64_t buffer[3] = {0};
        tag.GetBlob(blob_name, reinterpret_cast<char*>(buffer), data_size);

        REQUIRE(buffer[0] == data[0]);
        REQUIRE(buffer[1] == data[1]);
        REQUIRE(buffer[2] == data[2]);
    }
}

TEST_CASE("CTE blob overwriting", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_overwrite");

    SECTION("Can overwrite existing blob") {
        const char* blob_name = "overwrite_blob";
        const char* data1 = "First version";
        const char* data2 = "Second version - longer!";

        size_t size1 = strlen(data1) + 1;
        size_t size2 = strlen(data2) + 1;

        // Write first version
        tag.PutBlob(blob_name, data1, size1);

        // Overwrite with second version
        tag.PutBlob(blob_name, data2, size2);

        // Read and verify
        char buffer[100] = {0};
        tag.GetBlob(blob_name, buffer, size2);

        REQUIRE(strcmp(buffer, data2) == 0);
    }
}

TEST_CASE("CTE offset-based operations", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_offset");

    SECTION("Can write to different offsets in same blob") {
        const char* blob_name = "offset_blob";
        const char* part1 = "AAAA";
        const char* part2 = "BBBB";
        const char* part3 = "CCCC";

        size_t part_size = 4;

        // Write at offset 0
        tag.PutBlob(blob_name, part1, part_size, 0);

        // Write at offset 4
        tag.PutBlob(blob_name, part2, part_size, 4);

        // Write at offset 8
        tag.PutBlob(blob_name, part3, part_size, 8);

        // Read entire blob
        char buffer[13] = {0};
        tag.GetBlob(blob_name, buffer, 12, 0);

        REQUIRE(strncmp(buffer, "AAAABBBBCCCC", 12) == 0);
    }
}

TEST_CASE("CTE GetBlobSize operation", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_size");

    SECTION("GetBlobSize returns correct size for existing blob") {
        const char* blob_name = "sized_blob";
        const char* data = "Test data for size check";
        size_t expected_size = strlen(data) + 1;

        tag.PutBlob(blob_name, data, expected_size);

        chi::u64 actual_size = tag.GetBlobSize(blob_name);
        REQUIRE(actual_size >= expected_size);
    }

    SECTION("GetBlobSize updates after writes at different offsets") {
        const char* blob_name = "growing_blob";
        const char* data = "DATA";

        // Write at offset 0
        tag.PutBlob(blob_name, data, 4, 0);
        chi::u64 size1 = tag.GetBlobSize(blob_name);
        REQUIRE(size1 >= 4);

        // Write at offset 100 (creates gap)
        tag.PutBlob(blob_name, data, 4, 100);
        chi::u64 size2 = tag.GetBlobSize(blob_name);
        REQUIRE(size2 >= 104);
    }
}

TEST_CASE("CTE GetContainedBlobs operation", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_list");

    SECTION("GetContainedBlobs lists all blobs in tag") {
        // Create several blobs
        const char* names[] = {"blob_a", "blob_b", "blob_c"};
        const char* data = "x";

        for (int i = 0; i < 3; i++) {
            tag.PutBlob(names[i], data, 1);
        }

        // Get list of blobs
        std::vector<std::string> blob_list = tag.GetContainedBlobs();

        REQUIRE(blob_list.size() >= 3);

        // Verify all our blobs are in the list
        bool found_a = false, found_b = false, found_c = false;
        for (const auto& name : blob_list) {
            if (name == "blob_a") found_a = true;
            if (name == "blob_b") found_b = true;
            if (name == "blob_c") found_c = true;
        }

        REQUIRE(found_a);
        REQUIRE(found_b);
        REQUIRE(found_c);
    }
}

TEST_CASE("CTE large blob operations", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_large");

    SECTION("Can write and read large blobs") {
        const char* blob_name = "large_blob";
        const size_t large_size = 1024 * 1024; // 1 MB

        // Create large data buffer with pattern
        std::vector<uint8_t> data(large_size);
        for (size_t i = 0; i < large_size; i++) {
            data[i] = static_cast<uint8_t>(i % 256);
        }

        // Write large blob
        tag.PutBlob(blob_name, reinterpret_cast<const char*>(data.data()), large_size);

        // Read back
        std::vector<uint8_t> buffer(large_size);
        tag.GetBlob(blob_name, reinterpret_cast<char*>(buffer.data()), large_size);

        // Verify data integrity
        bool data_matches = true;
        for (size_t i = 0; i < large_size; i++) {
            if (buffer[i] != data[i]) {
                data_matches = false;
                break;
            }
        }

        REQUIRE(data_matches);
    }
}

TEST_CASE("CTE partial read operations", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_partial");

    SECTION("Can read partial blob data") {
        const char* blob_name = "partial_blob";
        const char* full_data = "0123456789ABCDEFGHIJ";
        size_t full_size = strlen(full_data);

        // Write full data
        tag.PutBlob(blob_name, full_data, full_size);

        // Read middle portion (offset 5, length 10)
        char buffer[11] = {0};
        tag.GetBlob(blob_name, buffer, 10, 5);

        REQUIRE(strncmp(buffer, "56789ABCDE", 10) == 0);
    }

    SECTION("Can read from specific offset") {
        const char* blob_name = "offset_read_blob";
        const char* data = "ABCDEFGHIJ";
        size_t data_size = 10;

        tag.PutBlob(blob_name, data, data_size);

        // Read last 5 bytes
        char buffer[6] = {0};
        tag.GetBlob(blob_name, buffer, 5, 5);

        REQUIRE(strcmp(buffer, "FGHIJ") == 0);
    }
}

TEST_CASE("CTE multiple tags", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    SECTION("Different tags maintain separate blob namespaces") {
        wrp_cte::core::Tag tag1("test_tag_multi_1");
        wrp_cte::core::Tag tag2("test_tag_multi_2");

        const char* blob_name = "shared_name";
        const char* data1 = "Tag 1 data";
        const char* data2 = "Tag 2 data";

        // Write different data to same blob name in different tags
        tag1.PutBlob(blob_name, data1, strlen(data1) + 1);
        tag2.PutBlob(blob_name, data2, strlen(data2) + 1);

        // Read from each tag
        char buffer1[20] = {0};
        char buffer2[20] = {0};
        tag1.GetBlob(blob_name, buffer1, strlen(data1) + 1);
        tag2.GetBlob(blob_name, buffer2, strlen(data2) + 1);

        REQUIRE(strcmp(buffer1, data1) == 0);
        REQUIRE(strcmp(buffer2, data2) == 0);
    }
}

#include <catch2/catch_test_macros.hpp>
#include "../../common/cte_runtime.h"
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

TEST_CASE("CTE Tag creation and basic PutBlob/GetBlob", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    SECTION("Can create a Tag and write/read simple data") {
        wrp_cte::core::Tag tag("test_tag_basic");
        const char* blob_name = "simple_blob";
        const char* data = "Hello, CTE!";
        size_t data_size = strlen(data) + 1;

        tag.PutBlob(blob_name, data, data_size);

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

        tag.PutBlob(blob_name, reinterpret_cast<const char*>(data), data_size);

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

        tag.PutBlob(blob_name, data1, size1);
        tag.PutBlob(blob_name, data2, size2);

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

        tag.PutBlob(blob_name, part1, part_size, 0);
        tag.PutBlob(blob_name, part2, part_size, 4);
        tag.PutBlob(blob_name, part3, part_size, 8);

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

        tag.PutBlob(blob_name, data, 4, 0);
        chi::u64 size1 = tag.GetBlobSize(blob_name);
        REQUIRE(size1 >= 4);

        tag.PutBlob(blob_name, data, 4, 100);
        chi::u64 size2 = tag.GetBlobSize(blob_name);
        REQUIRE(size2 >= 104);
    }
}

TEST_CASE("CTE GetContainedBlobs operation", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_list");

    SECTION("GetContainedBlobs lists all blobs in tag") {
        const char* names[] = {"blob_a", "blob_b", "blob_c"};
        const char* data = "x";

        for (int i = 0; i < 3; i++) {
            tag.PutBlob(names[i], data, 1);
        }

        std::vector<std::string> blob_list = tag.GetContainedBlobs();

        REQUIRE(blob_list.size() >= 3);

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

        std::vector<uint8_t> data(large_size);
        for (size_t i = 0; i < large_size; i++) {
            data[i] = static_cast<uint8_t>(i % 256);
        }

        tag.PutBlob(blob_name, reinterpret_cast<const char*>(data.data()), large_size);

        std::vector<uint8_t> buffer(large_size);
        tag.GetBlob(blob_name, reinterpret_cast<char*>(buffer.data()), large_size);

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

        tag.PutBlob(blob_name, full_data, full_size);

        char buffer[11] = {0};
        tag.GetBlob(blob_name, buffer, 10, 5);

        REQUIRE(strncmp(buffer, "56789ABCDE", 10) == 0);
    }

    SECTION("Can read from specific offset") {
        const char* blob_name = "offset_read_blob";
        const char* data = "ABCDEFGHIJ";
        size_t data_size = 10;

        tag.PutBlob(blob_name, data, data_size);

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

        tag1.PutBlob(blob_name, data1, strlen(data1) + 1);
        tag2.PutBlob(blob_name, data2, strlen(data2) + 1);

        char buffer1[20] = {0};
        char buffer2[20] = {0};
        tag1.GetBlob(blob_name, buffer1, strlen(data1) + 1);
        tag2.GetBlob(blob_name, buffer2, strlen(data2) + 1);

        REQUIRE(strcmp(buffer1, data1) == 0);
        REQUIRE(strcmp(buffer2, data2) == 0);
    }
}

TEST_CASE("CTE overwrite with smaller data", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_overwrite_smaller");

    SECTION("GetBlobSize after overwriting with smaller data") {
        const char* blob_name = "shrink_blob";
        const char* large_data = "This is a longer string!!";
        const char* small_data = "Short";

        size_t large_size = strlen(large_data) + 1;  // 26 bytes
        size_t small_size = strlen(small_data) + 1;   // 6 bytes

        tag.PutBlob(blob_name, large_data, large_size);
        chi::u64 size_after_large = tag.GetBlobSize(blob_name);

        tag.PutBlob(blob_name, small_data, small_size, 0);
        chi::u64 size_after_small = tag.GetBlobSize(blob_name);

        // Document whether CTE truncates or preserves the old size.
        // If size_after_small == large_size, CTE does NOT truncate,
        // meaning CteBlobStore must delete-then-put for overwrites.
        INFO("Size after large write: " << size_after_large);
        INFO("Size after small overwrite: " << size_after_small);

        // We expect CTE does NOT truncate (size stays at large_size).
        // If this FAILS, CTE truncates and CteBlobStore can skip delete-before-put.
        CHECK(size_after_small >= large_size);
    }
}

TEST_CASE("CTE GetBlobSize on non-existent blob", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_nonexist_size");

    SECTION("GetBlobSize behavior for blob that was never created") {
        // This informs our Exists() implementation.
        // Possible outcomes:
        //   - throws std::runtime_error → use try/catch for Exists
        //   - returns 0 → check for 0
        //   - returns some sentinel → check for sentinel
        bool threw = false;
        chi::u64 returned_size = 0;

        try {
            returned_size = tag.GetBlobSize("definitely_does_not_exist");
        } catch (const std::exception&) {
            threw = true;
        }

        INFO("Threw exception: " << threw);
        INFO("Returned size: " << returned_size);

        // Document the actual behavior — at least one of these should be true
        CHECK((threw || returned_size == 0));
    }
}

TEST_CASE("CTE empty blob name", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_empty_name");

    SECTION("PutBlob/GetBlob with empty string blob name") {
        // Does CTE accept "" as a blob name?
        bool put_threw = false;
        try {
            const char* data = "test";
            tag.PutBlob("", data, 4);
        } catch (const std::exception&) {
            put_threw = true;
        }

        INFO("PutBlob with empty name threw: " << put_threw);

        if (!put_threw) {
            // If put succeeded, can we read it back?
            char buffer[4] = {0};
            bool get_threw = false;
            try {
                tag.GetBlob("", buffer, 4);
            } catch (const std::exception&) {
                get_threw = true;
            }

            INFO("GetBlob with empty name threw: " << get_threw);
            if (!get_threw) {
                REQUIRE(strncmp(buffer, "test", 4) == 0);
            }
        }
    }
}

TEST_CASE("CTE delete blob via client", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_delete");

    SECTION("AsyncDelBlob removes blob and GetBlobSize reflects deletion") {
        const char* blob_name = "deletable_blob";
        const char* data = "delete me";
        size_t data_size = strlen(data) + 1;

        tag.PutBlob(blob_name, data, data_size);

        chi::u64 size_before = tag.GetBlobSize(blob_name);
        REQUIRE(size_before >= data_size);

        auto del_task = WRP_CTE_CLIENT->AsyncDelBlob(
            tag.GetTagId(), blob_name);
        del_task.Wait();

        bool threw = false;
        chi::u64 size_after = 0;
        try {
            size_after = tag.GetBlobSize(blob_name);
        } catch (const std::exception&) {
            threw = true;
        }

        INFO("GetBlobSize after delete threw: " << threw);
        INFO("GetBlobSize after delete returned: " << size_after);

        CHECK((threw || size_after == 0));
    }

    SECTION("Can re-create blob after deletion") {
        const char* blob_name = "reuse_blob";
        const char* data1 = "first";
        const char* data2 = "second";

        tag.PutBlob(blob_name, data1, strlen(data1) + 1);

        auto del_task = WRP_CTE_CLIENT->AsyncDelBlob(
            tag.GetTagId(), blob_name);
        del_task.Wait();

        tag.PutBlob(blob_name, data2, strlen(data2) + 1);

        char buffer[20] = {0};
        tag.GetBlob(blob_name, buffer, strlen(data2) + 1);
        REQUIRE(strcmp(buffer, data2) == 0);
    }
}

TEST_CASE("CTE zero-length data", "[integration][iowarp][cte]") {
    EnsureCteRuntime();
    wrp_cte::core::Tag tag("test_tag_zero_len");

    SECTION("PutBlob with zero-length data") {
        // CTE GetBlob validates data_size > 0, but what about PutBlob?
        bool put_threw = false;
        try {
            tag.PutBlob("zero_blob", "", 0);
        } catch (const std::exception&) {
            put_threw = true;
        }

        INFO("PutBlob with size=0 threw: " << put_threw);

        if (!put_threw) {
            chi::u64 size = tag.GetBlobSize("zero_blob");
            INFO("GetBlobSize for zero-length blob: " << size);
            CHECK(size == 0);
        }
    }
}

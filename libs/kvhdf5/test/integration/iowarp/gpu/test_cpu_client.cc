#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include <cstring>

// 4096-byte blobs give us room to exercise realistic I/O paths without
// hitting alignment or minimum-size edge cases.
static constexpr size_t kBlobSize = 4096;

TEST_CASE("CPU async AsyncPutBlob succeeds", "[integration][iowarp][cte_gpu]") {
    EnsureGpuCteRuntime();

    auto buf = CHI_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(buf.IsNull());
    std::memset(buf.ptr_, 0xAB, kBlobSize);

    hipc::ShmPtr<> blob_data = buf.shm_.template Cast<void>();
    auto put_task = WRP_CTE_CLIENT->AsyncPutBlob(
        g_gpu_cte_tag_id,
        "cpu_put_test_blob",
        0, kBlobSize,
        blob_data,
        1.0f,
        wrp_cte::core::Context(),
        0,
        chi::PoolQuery::Local());
    put_task.Wait();

    REQUIRE(put_task->GetReturnCode() == 0);
    CHI_IPC->FreeBuffer(buf);
}

TEST_CASE("CPU async AsyncGetBlob succeeds", "[integration][iowarp][cte_gpu]") {
    EnsureGpuCteRuntime();

    // put first so there is something to get
    {
        auto put_buf = CHI_IPC->AllocateBuffer(kBlobSize);
        REQUIRE_FALSE(put_buf.IsNull());
        std::memset(put_buf.ptr_, 0xCD, kBlobSize);
        hipc::ShmPtr<> put_shm = put_buf.shm_.template Cast<void>();
        auto put_task = WRP_CTE_CLIENT->AsyncPutBlob(
            g_gpu_cte_tag_id,
            "cpu_get_test_blob",
            0, kBlobSize,
            put_shm,
            1.0f,
            wrp_cte::core::Context(),
            0,
            chi::PoolQuery::Local());
        put_task.Wait();
        REQUIRE(put_task->GetReturnCode() == 0);
        CHI_IPC->FreeBuffer(put_buf);
    }

    auto get_buf = CHI_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(get_buf.IsNull());
    hipc::ShmPtr<> get_shm = get_buf.shm_.template Cast<void>();
    auto get_task = WRP_CTE_CLIENT->AsyncGetBlob(
        g_gpu_cte_tag_id,
        "cpu_get_test_blob",
        0, kBlobSize,
        0,
        get_shm,
        chi::PoolQuery::Local());
    get_task.Wait();

    REQUIRE(get_task->GetReturnCode() == 0);
    CHI_IPC->FreeBuffer(get_buf);
}

TEST_CASE("CPU async roundtrip data integrity", "[integration][iowarp][cte_gpu]") {
    EnsureGpuCteRuntime();

    // fill put buffer with a known pattern
    auto put_buf = CHI_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(put_buf.IsNull());
    auto *src = reinterpret_cast<unsigned char *>(put_buf.ptr_);
    for (size_t i = 0; i < kBlobSize; ++i) {
        src[i] = static_cast<unsigned char>(i % 251);
    }
    hipc::ShmPtr<> put_shm = put_buf.shm_.template Cast<void>();
    auto put_task = WRP_CTE_CLIENT->AsyncPutBlob(
        g_gpu_cte_tag_id,
        "cpu_roundtrip_blob",
        0, kBlobSize,
        put_shm,
        1.0f,
        wrp_cte::core::Context(),
        0,
        chi::PoolQuery::Local());
    put_task.Wait();
    REQUIRE(put_task->GetReturnCode() == 0);
    CHI_IPC->FreeBuffer(put_buf);

    // read it back and verify
    auto get_buf = CHI_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(get_buf.IsNull());
    std::memset(get_buf.ptr_, 0, kBlobSize);
    hipc::ShmPtr<> get_shm = get_buf.shm_.template Cast<void>();
    auto get_task = WRP_CTE_CLIENT->AsyncGetBlob(
        g_gpu_cte_tag_id,
        "cpu_roundtrip_blob",
        0, kBlobSize,
        0,
        get_shm,
        chi::PoolQuery::Local());
    get_task.Wait();
    REQUIRE(get_task->GetReturnCode() == 0);

    auto *dst = reinterpret_cast<unsigned char *>(get_buf.ptr_);
    bool match = true;
    for (size_t i = 0; i < kBlobSize; ++i) {
        if (dst[i] != static_cast<unsigned char>(i % 251)) {
            match = false;
            break;
        }
    }
    CHI_IPC->FreeBuffer(get_buf);
    REQUIRE(match);
}

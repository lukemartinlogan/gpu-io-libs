#if HSHM_ENABLE_CUDA

#include <catch2/catch_test_macros.hpp>
#include "gpu_cte_fixture.h"
#include "kvhdf5/gpu_cte_blob_store.h"
#include "kvhdf5/blob_store.h"
#include "kvhdf5/types/chunk_key.h"
#include <cstring>
#include <thread>
#include <chrono>

using namespace kvhdf5;

// ---------------------------------------------------------------------------
// TestPair: small POD struct for Test B.  Must be at file scope so that
// KVHDF5_AUTO_SERDE can specialize serde::SerializePOD<> for it.
// No non-trivial constructors — stays plain POD.
// ---------------------------------------------------------------------------

struct TestPair {
    uint32_t a;
    uint32_t b;
};

// Opt TestPair into the serde layer so BlobStore<B>::PutBlob/GetBlob's
// requires-clause (serde::SerializePOD<V>::value) is satisfied.
KVHDF5_AUTO_SERDE(TestPair);

// ---------------------------------------------------------------------------
// Result struct (same shape as the existing gpu_cte_blob_store_test.cu)
// ---------------------------------------------------------------------------

struct TypedBlobTestResult {
    int status;      // 1 = pass, negative = error code
    int data_match;  // 1 = data matched, 0 = mismatch
};

// ---------------------------------------------------------------------------
// Regression: direct-path 6-byte key roundtrip (bypasses BlobStore wrapper).
//
// Historical note: iowarp-core's GPU PutBlob handler builds a compound key
// via chi::priv::string::reserve(22 + blob_name_len). For hex-encoded keys
// of >= 6 bytes (>= 12 hex chars) this reserve crosses the 32-byte SSO
// boundary and forces a device-side heap allocation that faults with
// CUDA Error 700. GpuCteBlobStore works around this by emitting
// fixed-width 10-char hash-derived blob names (see KeyToName in
// gpu_cte_blob_store.h) that stay within SSO regardless of key length.
// This test exercises a 6-byte key end-to-end to guard that workaround.
// ---------------------------------------------------------------------------

__global__ void kernel_direct_6byte_key(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    cstd::array<byte_t, 6> key = {
        byte_t{0x11}, byte_t{0x22}, byte_t{0x33},
        byte_t{0x44}, byte_t{0x55}, byte_t{0x66}
    };
    cstd::array<byte_t, 4> value = {
        byte_t{0xAA}, byte_t{0xBB}, byte_t{0xCC}, byte_t{0xDD}
    };

    bool put_ok = store.PutBlob(key, value);
    if (!put_ok) { d_result->status = -1; return; }

    cstd::array<byte_t, 4> out;
    auto result = store.GetBlob(key, out);
    if (!result.has_value()) { d_result->status = -2; return; }
    if (result->size() != 4) { d_result->status = -3; return; }

    d_result->data_match = 1;
    for (int i = 0; i < 4; ++i) {
        if (out[i] != value[i]) { d_result->data_match = 0; d_result->status = -4; return; }
    }
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Test A — Put/Get POD roundtrip (uint64_t key, uint64_t value)
// ---------------------------------------------------------------------------

__global__ void kernel_typed_pod_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    BlobStore<GpuCteBlobStore> bs(&store);

    uint64_t key   = 0xDEADBEEFCAFEBABEULL;
    uint64_t value = 0x0123456789ABCDEFULL;

    bool put_ok = bs.PutBlob(key, value);
    if (!put_ok) { d_result->status = -1; return; }

    auto result = bs.GetBlob<uint64_t, uint64_t>(key);
    if (!result.has_value()) { d_result->status = -2; return; }

    if (*result != value) { d_result->status = -3; return; }

    d_result->data_match = 1;
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Test B — Put/Get small struct via bit_cast serde
// ---------------------------------------------------------------------------

__global__ void kernel_typed_struct_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    BlobStore<GpuCteBlobStore> bs(&store);

    uint64_t key = 0x0000000000000042ULL;
    TestPair value;
    value.a = 0xABCD1234U;
    value.b = 0x56789EFAU;

    bool put_ok = bs.PutBlob(key, value);
    if (!put_ok) { d_result->status = -1; return; }

    auto result = bs.GetBlob<uint64_t, TestPair>(key);
    if (!result.has_value()) { d_result->status = -2; return; }

    if (result->a != value.a) { d_result->status = -3; return; }
    if (result->b != value.b) { d_result->status = -4; return; }

    d_result->data_match = 1;
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Test C — PutRawBlob / GetRawBlob with a ChunkKey
// ---------------------------------------------------------------------------

__global__ void kernel_typed_raw_blob_roundtrip(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    BlobStore<GpuCteBlobStore> bs(&store);

    // Build a ChunkKey with dataset id=7, 2-D chunk at (3, 5).
    DatasetId dataset(uint64_t{7});
    cstd::array<uint64_t, 2> coords_arr = {uint64_t{3}, uint64_t{5}};
    cstd::span<const uint64_t> coords_span(coords_arr.data(), 2);
    ChunkKey key(dataset, coords_span);

    // 16-byte payload with a known pattern.
    cstd::array<byte_t, 16> payload = {
        byte_t{0x01}, byte_t{0x02}, byte_t{0x03}, byte_t{0x04},
        byte_t{0x05}, byte_t{0x06}, byte_t{0x07}, byte_t{0x08},
        byte_t{0x09}, byte_t{0x0A}, byte_t{0x0B}, byte_t{0x0C},
        byte_t{0x0D}, byte_t{0x0E}, byte_t{0x0F}, byte_t{0x10}
    };

    cstd::span<const byte_t> put_span(payload.data(), payload.size());
    bool put_ok = bs.PutRawBlob(key, put_span);
    if (!put_ok) { d_result->status = -1; return; }

    // Get back with exactly-sized buffer (avoids GetBlob size-mismatch bug).
    cstd::array<byte_t, 16> out_buf;
    cstd::span<byte_t> get_span(out_buf.data(), out_buf.size());
    auto result = bs.GetRawBlob(key, get_span);
    if (!result.has_value()) { d_result->status = -2; return; }
    if (result->size() != 16) { d_result->status = -3; return; }

    d_result->data_match = 1;
    for (size_t i = 0; i < 16; ++i) {
        if (out_buf[i] != payload[i]) {
            d_result->data_match = 0;
            d_result->status = -4;
            return;
        }
    }

    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Test D — Typed Exists
// ---------------------------------------------------------------------------

__global__ void kernel_typed_exists(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    BlobStore<GpuCteBlobStore> bs(&store);

    uint64_t key   = 0xFEEDFACEDEADC0DEULL;
    uint64_t value = 0x000000000000FFFFULL;

    // Key must not exist before any Put.
    if (bs.Exists(key)) { d_result->status = -1; return; }

    bool put_ok = bs.PutBlob(key, value);
    if (!put_ok) { d_result->status = -2; return; }

    // Key must exist after Put.
    if (!bs.Exists(key)) { d_result->status = -3; return; }

    d_result->data_match = 1;
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Test E — Typed Delete
// ---------------------------------------------------------------------------

__global__ void kernel_typed_delete(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    BlobStore<GpuCteBlobStore> bs(&store);

    uint64_t key   = 0x1122334455667788ULL;
    uint64_t value = 0xAABBCCDDEEFF0011ULL;

    bool put_ok = bs.PutBlob(key, value);
    if (!put_ok) { d_result->status = -1; return; }

    bool del_ok = bs.DeleteBlob(key);
    if (!del_ok) { d_result->status = -2; return; }

    // Get on deleted key must be absent.
    auto get_result = bs.GetBlob<uint64_t, uint64_t>(key);
    if (get_result.has_value()) { d_result->status = -3; return; }

    // Exists on deleted key must be false.
    if (bs.Exists(key)) { d_result->status = -4; return; }

    d_result->data_match = 1;
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Test F — Custom serializer / deserializer lambda overloads of PutBlob /
// GetBlob.  This is the exact code path Container uses: PutGroup builds an
// inline lambda that calls GroupMetadata::Serialize, GetGroup builds an
// inline lambda that calls GroupMetadata::Deserialize.  We exercise the
// lambda path directly by writing fields in a deliberately non-natural order
// and reading them back; if the lambda is somehow bypassed by the implicit
// bit-cast overload, the values would mismatch.
// ---------------------------------------------------------------------------

struct ReversedSerdeType {
    uint32_t a;
    uint16_t b;
    uint8_t  c;
};

__global__ void kernel_typed_custom_serde(
    chi::IpcManagerGpuInfo gpu_info,
    GpuCteBlobStore store,
    TypedBlobTestResult *d_result)
{
    d_result->status = 0;
    d_result->data_match = 0;
    CHIMAERA_GPU_INIT(gpu_info);
    if (threadIdx.x != 0) return;

    BlobStore<GpuCteBlobStore> bs(&store);

    uint64_t key = 0xC0FFEE00C0FFEE00ULL;
    ReversedSerdeType value = {0xDEADBEEFu, 0xCAFEu, 0xAAu};

    auto serialize_fn = [](serde::BufferReaderWriter& w, const ReversedSerdeType& v) {
        // Reversed field order: c, b, a.
        serde::Write(w, v.c);
        serde::Write(w, v.b);
        serde::Write(w, v.a);
    };
    auto deserialize_fn = [](serde::BufferDeserializer& r) -> ReversedSerdeType {
        ReversedSerdeType v;
        v.c = serde::Read<uint8_t>(r);
        v.b = serde::Read<uint16_t>(r);
        v.a = serde::Read<uint32_t>(r);
        return v;
    };

    bool put_ok = bs.PutBlob<uint64_t, ReversedSerdeType>(key, value, serialize_fn);
    if (!put_ok) { d_result->status = -1; return; }

    auto result = bs.GetBlob<uint64_t, ReversedSerdeType>(key, deserialize_fn);
    if (!result.has_value())     { d_result->status = -2; return; }
    if (result->a != value.a)    { d_result->status = -3; return; }
    if (result->b != value.b)    { d_result->status = -4; return; }
    if (result->c != value.c)    { d_result->status = -5; return; }

    d_result->data_match = 1;
    d_result->status = 1;
}

// ---------------------------------------------------------------------------
// Helper: launch a typed-blob-store kernel and poll for completion.
// Named RunTypedBlobTest to stay distinct from RunGpuBlobTest in the
// existing gpu_cte_blob_store_test.cu — do NOT rename/merge the two.
// ---------------------------------------------------------------------------

static TypedBlobTestResult RunTypedBlobTest(
    void (*kernel)(chi::IpcManagerGpuInfo, GpuCteBlobStore, TypedBlobTestResult*),
    GpuCteBlobStore store)
{
    auto *gpu_ipc = CHI_CPU_IPC->GetGpuIpcManager();
    chi::IpcManagerGpuInfo gpu_info = gpu_ipc->GetClientGpuInfo(0);

    gpu_ipc->PauseGpuOrchestrator();

    volatile TypedBlobTestResult *d_result;
    cudaMallocHost(const_cast<TypedBlobTestResult **>(&d_result),
                   sizeof(TypedBlobTestResult));
    d_result->status = 0;
    d_result->data_match = 0;

    cudaGetLastError();
    void *stream = hshm::GpuApi::CreateStream();
    kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        gpu_info, store, const_cast<TypedBlobTestResult *>(d_result));

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        gpu_ipc->ResumeGpuOrchestrator();
        hshm::GpuApi::DestroyStream(stream);
        return {-201, 0};
    }

    // Resume AFTER launch so the orchestrator can service CTE tasks.
    gpu_ipc->ResumeGpuOrchestrator();

    // Poll pinned memory — never cudaStreamSynchronize (deadlocks persistent kernel).
    constexpr int kTimeoutUs = 30 * 1000 * 1000;  // 30 s
    int elapsed_us = 0;
    while (d_result->status == 0 && elapsed_us < kTimeoutUs) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        elapsed_us += 100;
    }

    TypedBlobTestResult result{d_result->status, d_result->data_match};
    if (result.status == 0) {
        result.status = -300;  // timeout sentinel
    }

    // Intentionally leak pinned buffer and stream: freeing them can block
    // against the persistent GPU orchestrator kernel.
    return result;
}

// ---------------------------------------------------------------------------
// Catch2 TEST_CASEs (host-only, hidden from device compilation pass)
// ---------------------------------------------------------------------------

#if !HSHM_IS_GPU

TEST_CASE("GpuCteBlobStore - direct-path 6-byte key roundtrip",
          "[gpu_blob_store_direct_6byte]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_blob_store_direct_6byte");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_direct_6byte_key, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("BlobStore<GpuCteBlobStore> - Put/Get POD roundtrip from kernel",
          "[gpu_typed_blob_store_pod]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_typed_pod_roundtrip");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_typed_pod_roundtrip, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("BlobStore<GpuCteBlobStore> - Put/Get small struct (bit_cast serde) from kernel",
          "[gpu_typed_blob_store_struct]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_typed_struct_roundtrip");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_typed_struct_roundtrip, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("BlobStore<GpuCteBlobStore> - PutRawBlob/GetRawBlob with ChunkKey from kernel",
          "[gpu_typed_blob_store_raw_blob]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_typed_raw_blob_roundtrip");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_typed_raw_blob_roundtrip, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("BlobStore<GpuCteBlobStore> - Typed Exists from kernel",
          "[gpu_typed_blob_store_exists]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_typed_exists");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_typed_exists, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("BlobStore<GpuCteBlobStore> - Typed Delete from kernel",
          "[gpu_typed_blob_store_delete]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_typed_delete");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_typed_delete, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

TEST_CASE("BlobStore<GpuCteBlobStore> - Custom serializer/deserializer lambdas from kernel",
          "[gpu_typed_blob_store_custom_serde]") {
    EnsureGpuCteRuntime();
    chi::PoolId pool_id = g_gpu_cte_pool_id;

    auto tag_task = g_gpu_cte_client->AsyncGetOrCreateTag("gpu_typed_custom_serde");
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    GpuCteBlobStore store = GpuCteBlobStore::Create(tag_id, pool_id);
    REQUIRE(store.IsValid());
    auto result = RunTypedBlobTest(kernel_typed_custom_serde, store);
    store.Destroy();

    INFO("kernel status: " << result.status);
    REQUIRE(result.status == 1);
    REQUIRE(result.data_match == 1);
}

#endif  // !HSHM_IS_GPU

#endif  // HSHM_ENABLE_CUDA

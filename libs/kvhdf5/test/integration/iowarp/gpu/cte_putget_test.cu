/*
 * iowarp CTE GPU integration test — PutBlob + GetBlob round trip.
 *
 * Asserts the iowarp "producer-only" GPU CTE contract that kvhdf5 is built on:
 * the host pre-allocates the Task + co-located FutureShm + blob buffer in
 * registered kDeviceMem backends, and the kernel only Send()s the task and
 * Wait()s on its future — no orchestrator pause/resume, no device-side
 * allocation. If a future iowarp bump breaks this contract, this test fails
 * and the failure is attributable to iowarp, not to kvhdf5.
 *
 * Faithfully mirrors iowarp-core's reference proof-of-concept
 *   context-transfer-engine/test/unit/gpu/test_cte_devmem_putget.cc
 * swapping only the harness (SimpleTest -> Catch2, the kvhdf5 standard) and
 * lifting the one-time bring-up into a shared singleton (SharedCteEnv) so that
 * additional GPU integration cases added later reuse the expensive server
 * start-up instead of re-paying it per case.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/future.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

// Catch2 macros (REQUIRE) are used by host-only code below (the shared
// environment + the TEST_CASE). Only pull them in on the host compilation
// pass; the nvcc device pass never sees REQUIRE.
#if !CTP_IS_DEVICE_PASS
#include <catch2/catch_test_macros.hpp>
#endif

using namespace std::chrono_literals;

namespace cte = clio::cte::core;

namespace {

constexpr chi::u32 kBlobBytes = 256;
constexpr chi::u32 kPatternSeed = 0xC3u;
constexpr const char *kBlobName = "kdev";  // SSO-friendly (<= 23 chars)

#if !CTP_IS_DEVICE_PASS

/**
 * One-time Chimaera bring-up shared by every TEST_CASE in this executable.
 *
 * Server start-up + CTE pool create + bdev registration is expensive, and the
 * Chimaera server is process-global, so it must happen exactly once. A
 * function-local static (see SharedCteEnv) gives lazy, once-only construction
 * the first time any test touches it; Catch2 runs all cases in a single
 * process, so they all share this same environment. Bring-up failures throw,
 * which Catch2 reports as a failed test rather than crashing the run.
 */
struct IowarpCteEnv {
  cte::TagId tag_id;

  IowarpCteEnv() {
    std::fprintf(stderr, "[init] bringing up Chimaera server (once)\n");
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer))
      throw std::runtime_error("CHIMAERA_INIT(kServer) failed");
    if (!cte::CLIO_CTE_CLIENT_INIT())
      throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");

    auto *cte_client = CLIO_CTE_CLIENT;
    if (cte_client == nullptr)
      throw std::runtime_error("CLIO_CTE_CLIENT is null");
    cte_client->Init(cte::kCtePoolId);

    cte::CreateParams params;
    auto create_task = cte_client->AsyncCreate(
        chi::PoolQuery::Dynamic(), cte::kCtePoolName, cte::kCtePoolId, params);
    create_task.Wait();
    if (create_task->GetReturnCode() != 0)
      throw std::runtime_error("CTE pool AsyncCreate failed");
    std::this_thread::sleep_for(50ms);

    // Register a kRam bdev target so PutBlob has somewhere to land.
    const chi::u64 kRamCapacity = 64ULL << 20;  // 64 MiB
    chi::PoolId bdev_pool_id(960, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto bdev_create = bdev_client.AsyncCreate(
        chi::PoolQuery::Dynamic(), std::string("kvhdf5_itest_ram"),
        bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
    bdev_create.Wait();
    if (bdev_create->GetReturnCode() != 0)
      throw std::runtime_error("bdev AsyncCreate failed");
    std::this_thread::sleep_for(50ms);

    auto reg_task = cte_client->AsyncRegisterTarget(
        "kvhdf5_itest_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
        chi::PoolQuery::Local(), bdev_pool_id);
    reg_task.Wait();
    if (reg_task->GetReturnCode() != 0)
      throw std::runtime_error("AsyncRegisterTarget failed");
    std::this_thread::sleep_for(50ms);

    auto tag_task = cte_client->AsyncGetOrCreateTag("kvhdf5_itest_tag");
    tag_task.Wait();
    if (tag_task->GetReturnCode() != 0)
      throw std::runtime_error("AsyncGetOrCreateTag failed");
    tag_id = tag_task->tag_id_;

    std::fprintf(stderr, "[init] ready (tag=(%u,%u))\n", tag_id.major_,
                 tag_id.minor_);
  }
};

/** Lazily construct (once) and return the shared Chimaera/CTE environment. */
IowarpCteEnv &SharedCteEnv() {
  static IowarpCteEnv env;
  return env;
}

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace

/** Fill the blob_data buffer on device with the byte pattern. */
__global__ void CteFillKernel(char *buf, chi::u32 size, chi::u32 seed) {
  chi::u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  buf[i] = static_cast<char>((seed ^ i) & 0xFFu);
}

/** Submit one pre-built device-resident task and wait for completion. */
__global__ void CteSubmitPutKernel(chi::IpcManagerGpuInfo info,
                                   ctp::ipc::FullPtr<cte::PutBlobTask> task) {
  CHIMAERA_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  auto fut = g_ipc_manager_ptr->Send(task);
  fut.Wait();
  (void)g_ipc_manager;
}

__global__ void CteSubmitGetKernel(chi::IpcManagerGpuInfo info,
                                   ctp::ipc::FullPtr<cte::GetBlobTask> task) {
  CHIMAERA_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  auto fut = g_ipc_manager_ptr->Send(task);
  fut.Wait();
  (void)g_ipc_manager;
}

/** Verify the byte pattern on the device-resident GET buffer (host reads it
 *  back via cudaMemcpy after the kernel returns). */
chi::u32 VerifyDevicePattern(const char *device_buf, chi::u32 size,
                             chi::u32 seed) {
  std::vector<char> host(size);
  ctp::GpuApi::Memcpy(host.data(), device_buf, size);
  for (chi::u32 i = 0; i < size; ++i) {
    char want = static_cast<char>((seed ^ i) & 0xFFu);
    if (host[i] != want) return i;
  }
  return size;  // all match
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("iowarp CTE PutBlob+GetBlob round trip (device-memory task & data)",
          "[integration][gpu][cte][putget]") {
  auto &env = SharedCteEnv();
  auto *ipc = CLIO_CPU_IPC;
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() >= 1u);

  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(/*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

  // ---- 1) Allocate kDeviceMem backends ----
  // (a) Task slots: PutBlobTask + FutureShm + GetBlobTask + FutureShm.
  const chi::u32 kPutSlot =
      sizeof(cte::PutBlobTask) + sizeof(chi::gpu::FutureShm);
  const chi::u32 kGetSlot =
      sizeof(cte::GetBlobTask) + sizeof(chi::gpu::FutureShm);
  const chi::u32 kTaskBackendBytes = kPutSlot + kGetSlot + 64;
  char *task_dev_base = nullptr;
  auto task_alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, chi::gpu::IpcManager::MemKind::kDeviceMem,
      kTaskBackendBytes, &task_dev_base);
  REQUIRE(!task_alloc_id.IsNull());
  REQUIRE(task_dev_base != nullptr);

  // (b) Blob data: a single 256-byte buffer used for both Put and Get.
  char *blob_dev = nullptr;
  auto blob_alloc_id = ipc->AllocateAndRegisterGpuBackend(
      /*gpu_id=*/0, chi::gpu::IpcManager::MemKind::kDeviceMem, kBlobBytes,
      &blob_dev);
  REQUIRE(!blob_alloc_id.IsNull());
  REQUIRE(blob_dev != nullptr);

  // ---- 2) Construct host prototypes via placement-new and stamp them onto
  //         the device task slots via cudaMemcpy. ----
  // PutBlob prototype:
  alignas(64) char put_proto[kPutSlot];
  std::memset(put_proto, 0, sizeof(put_proto));
  ctp::ipc::ShmPtr<> put_blob_shm;
  put_blob_shm.alloc_id_.SetNull();
  put_blob_shm.off_ = reinterpret_cast<chi::u64>(blob_dev);
  auto *put_proto_task = new (put_proto) cte::PutBlobTask(
      chi::CreateTaskId(), cte::kCtePoolId, chi::PoolQuery::ToLocalCpu(),
      env.tag_id, kBlobName, /*offset=*/chi::u64(0),
      static_cast<chi::u64>(kBlobBytes), put_blob_shm,
      /*score=*/-1.0f, cte::Context(), /*flags=*/chi::u32(0));
  put_proto_task->pod_size_ = sizeof(cte::PutBlobTask);
  new (put_proto + sizeof(cte::PutBlobTask)) chi::gpu::FutureShm();
  ctp::GpuApi::Memcpy(task_dev_base, put_proto, sizeof(put_proto));

  // GetBlob prototype:
  alignas(64) char get_proto[kGetSlot];
  std::memset(get_proto, 0, sizeof(get_proto));
  ctp::ipc::ShmPtr<> get_blob_shm;
  get_blob_shm.alloc_id_.SetNull();
  get_blob_shm.off_ = reinterpret_cast<chi::u64>(blob_dev);
  auto *get_proto_task = new (get_proto) cte::GetBlobTask(
      chi::CreateTaskId(), cte::kCtePoolId, chi::PoolQuery::ToLocalCpu(),
      env.tag_id, kBlobName, /*offset=*/chi::u64(0),
      static_cast<chi::u64>(kBlobBytes), /*flags=*/chi::u32(0), get_blob_shm);
  get_proto_task->pod_size_ = sizeof(cte::GetBlobTask);
  new (get_proto + sizeof(cte::GetBlobTask)) chi::gpu::FutureShm();
  ctp::GpuApi::Memcpy(task_dev_base + kPutSlot, get_proto, sizeof(get_proto));

  // ---- 3) Fill blob_data on device with the source pattern. ----
  chi::u32 fill_threads = 256;
  chi::u32 fill_blocks = (kBlobBytes + fill_threads - 1) / fill_threads;
  CteFillKernel<<<fill_blocks, fill_threads>>>(blob_dev, kBlobBytes,
                                               kPatternSeed);
  ctp::GpuApi::Synchronize();

  // ---- 4) Build kernel-visible FullPtrs (raw device addresses stashed in
  //         off_, null alloc_id). ----
  ctp::ipc::FullPtr<cte::PutBlobTask> put_fp;
  put_fp.shm_.alloc_id_.SetNull();
  put_fp.shm_.off_ = reinterpret_cast<chi::u64>(task_dev_base);
  put_fp.ptr_ = reinterpret_cast<cte::PutBlobTask *>(task_dev_base);
  ctp::ipc::FullPtr<cte::GetBlobTask> get_fp;
  get_fp.shm_.alloc_id_.SetNull();
  get_fp.shm_.off_ = reinterpret_cast<chi::u64>(task_dev_base + kPutSlot);
  get_fp.ptr_ = reinterpret_cast<cte::GetBlobTask *>(task_dev_base + kPutSlot);

  // ---- 5) Launch the PutBlob kernel and wait. ----
  std::fprintf(stderr, "[put] launching CteSubmitPutKernel\n");
  CteSubmitPutKernel<<<1, 32>>>(gpu_info, put_fp);
  ctp::GpuApi::Synchronize();

  // Pull return_code_ back from device.
  cte::PutBlobTask put_after{};
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&put_after), task_dev_base,
                      sizeof(cte::PutBlobTask));
  std::fprintf(stderr, "[put] return_code=%u\n", put_after.return_code_.load());
  REQUIRE(put_after.return_code_.load() == 0u);

  // ---- 6) Zero out the blob_data buffer on device so the GetBlob readback
  //         is verifiable. ----
  std::vector<char> zeros(kBlobBytes, 0);
  ctp::GpuApi::Memcpy(blob_dev, zeros.data(), kBlobBytes);

  // ---- 7) Launch the GetBlob kernel and wait. ----
  std::fprintf(stderr, "[get] launching CteSubmitGetKernel\n");
  CteSubmitGetKernel<<<1, 32>>>(gpu_info, get_fp);
  ctp::GpuApi::Synchronize();

  cte::GetBlobTask get_after{};
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&get_after),
                      task_dev_base + kPutSlot, sizeof(cte::GetBlobTask));
  std::fprintf(stderr, "[get] return_code=%u\n", get_after.return_code_.load());
  REQUIRE(get_after.return_code_.load() == 0u);

  // ---- 8) Verify the device buffer contains the original pattern. ----
  chi::u32 first_bad = VerifyDevicePattern(blob_dev, kBlobBytes, kPatternSeed);
  if (first_bad != kBlobBytes) {
    std::fprintf(stderr, "[verify] mismatch at index %u (out of %u)\n",
                 first_bad, kBlobBytes);
  }
  REQUIRE(first_bad == kBlobBytes);
  std::fprintf(stderr, "[ok] cte devmem put+get round trip ok (%u bytes)\n",
               kBlobBytes);

  // ---- 9) Free backends. ----
  ipc->FreeGpuBackend(/*gpu_id=*/0, blob_alloc_id);
  ipc->FreeGpuBackend(/*gpu_id=*/0, task_alloc_id);
}

#endif  // !CTP_IS_DEVICE_PASS

#else

// Non-GPU build: nothing to test here.

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

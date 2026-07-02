#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // expose O_DIRECT from <fcntl.h> (raw-arm cache-bypass parity)
#endif
/*
 * THREE-WAY Gray-Scott I/O benchmark (the paper figure the advisor asked for):
 * ONE shared computation, THREE storage paths, wall-clock compared:
 *
 *   - raw   : no CLIO. GPU compute -> D2H to pinned host -> pwrite + fsync to disk.
 *   - sync  : CLIO, fused submit-AND-WAIT snapshot (GPU blocks on each PutBlob).
 *   - async : CLIO, fire-all snapshot + drain-at-end (server I/O overlaps GPU compute).
 *
 * SAME COMPUTATION across all three by construction: every arm runs the identical
 * `GsStepKernel` (one thread per cell) through the identical timed `RunSim` loop;
 * only the per-snapshot "sink" differs. So the comparison isolates the I/O + storage
 * backend, not the math.
 *
 * ROUTING AROUND THE iowarp ~16-large-backend ceiling (ADVISOR-REPORT §6a; the limit
 * is by COUNT not bytes, and dataset REUSE hangs — see refire probe): each arm reaches
 * the ~2 GB target with FEWER, BIGGER snapshots (<=~12 distinct datasets, each large),
 * keeping the proven fresh-dataset-per-snapshot recipe. The sync and async arms run in
 * SEPARATE processes (one arm per TEST_CASE invocation), so neither exceeds the ceiling.
 *
 * FAIR STORAGE: the CLIO arms can target a kFile bdev (O_DIRECT to a real file, cache-
 * bypassing) to match the raw arm's disk, or a kRam bdev for a RAM baseline — selected
 * by GSBENCH_BDEV. The raw arm always writes real files + fsync.
 *
 * All knobs are ENV VARS so scaling to 2 GB / switching to disk needs NO recompile:
 *   GSBENCH_N            grid dim (NxN float32)              default 512
 *   GSBENCH_CHUNKS       chunks per snapshot dataset         default 4
 *   GSBENCH_SNAPS        number of snapshots (<= ~12!)       default 4
 *   GSBENCH_STEPS_PER    sim steps between snapshots         default 8
 *   GSBENCH_BDEV         ram | file  (CLIO arms)             default ram
 *   GSBENCH_BDEV_CAP_MB  bdev capacity (MB)                  default 512
 *   GSBENCH_BDEV_PATH    kFile path (CLIO arms)              default ./gsbench_bdev.dat
 *   GSBENCH_DISK_DIR     raw-arm output dir                  default ./gsbench_raw_out
 *
 * Each arm prints ONE machine-parseable RESULT line (ms, MB, MB/s, checksum). A wrapper
 * runs all three processes and builds the relative table; the shared checksum proves all
 * three computed identical bytes. Cases HIDDEN ([.]); CLIO arms RUN AT num_threads=1.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/singletons.h>
#include <clio_ctp/util/gpu_api.h>

#include <kvhdf5/cpu_dataset.h>      // Layout
#include <kvhdf5/gpu_cte_dataset.h>
#include <kvhdf5/tag_path.h>         // CanonicalTag

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !CTP_IS_DEVICE_PASS
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#ifndef O_DIRECT
#define O_DIRECT 040000  // Linux x86/arm value; fallback if _GNU_SOURCE didn't expose it
#endif
#endif

using kvhdf5::byte_t;

namespace {
struct GsParams { float Du, Dv, F, k, dt; };
}  // namespace

// ---- shared computation: identical for every arm ---------------------------

// One Gray-Scott step, one thread per cell, periodic BCs. Pure CUDA.
__global__ void GsStepKernel(const float* u, const float* v, float* un, float* vn,
                             GsParams p, unsigned N) {
    unsigned cells = N * N;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cells) return;
    unsigned x = gid % N, y = gid / N;
    unsigned xm = (x == 0) ? (N - 1) : (x - 1);
    unsigned xp = (x == N - 1) ? 0u : (x + 1);
    unsigned ym = (y == 0) ? (N - 1) : (y - 1);
    unsigned yp = (y == N - 1) ? 0u : (y + 1);
    float uc = u[gid], vc = v[gid];
    float lap_u = u[y*N+xm] + u[y*N+xp] + u[ym*N+x] + u[yp*N+x] - 4.f*uc;
    float lap_v = v[y*N+xm] + v[y*N+xp] + v[ym*N+x] + v[yp*N+x] - 4.f*vc;
    float uvv = uc * vc * vc;
    un[gid] = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
    vn[gid] = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

// ---- CLIO snapshot kernels (device-facing handle) --------------------------

__device__ inline void CopyChunk(kvhdf5::GpuDatasetHandle& h, uint32_t c,
                                 const byte_t* src) {
    byte_t* dst = h.Data(c);
    uint64_t n = h.Size(c);
    uint64_t off = static_cast<uint64_t>(c) * n;
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[off + i];
}

// SYNC: copy each chunk then fused Write(c) = submit-AND-WAIT (GPU blocks).
__global__ void TwSnapSyncKernel(kvhdf5::GpuDatasetHandle h, const float* src) {
    CHIMAERA_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const byte_t* s = reinterpret_cast<const byte_t*>(src);
    for (uint32_t c = 0; c < h.Count(); ++c) {
        CopyChunk(h, c, s);
        __threadfence_system();
        __syncthreads();
        h.Write(c);
        __syncthreads();
    }
}

// ASYNC FIRE only: copy + fire every chunk's PutBlob, no wait. Drained later so the
// puts run on the server WHILE the subsequent sim steps run on the GPU.
__global__ void TwSnapFireKernel(kvhdf5::GpuDatasetHandle h, const float* src) {
    CHIMAERA_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    const byte_t* s = reinterpret_cast<const byte_t*>(src);
    for (uint32_t c = 0; c < h.Count(); ++c) {
        CopyChunk(h, c, s);
        __threadfence_system();
        __syncthreads();
        h.WriteAsync(c);
        __syncthreads();
    }
}

// Drain a fired snapshot's outstanding puts.
__global__ void TwDrainKernel(kvhdf5::GpuDatasetHandle h) {
    CHIMAERA_GPU_INIT(h.info_, /*ipc_ptr=*/nullptr);
    (void)g_ipc_manager;
    for (uint32_t c = 0; c < h.Count(); ++c) { h.WriteWait(c); __syncthreads(); }
}

#if !CTP_IS_DEVICE_PASS

namespace {

constexpr GsParams kGs{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};

// ---- env-var config --------------------------------------------------------

unsigned EnvU(const char* k, unsigned dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    long x = std::strtol(v, nullptr, 10);
    return x > 0 ? static_cast<unsigned>(x) : dflt;
}
std::string EnvS(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

struct Cfg {
    unsigned N       = EnvU("GSBENCH_N", 512);
    unsigned chunks  = EnvU("GSBENCH_CHUNKS", 4);
    unsigned snaps   = EnvU("GSBENCH_SNAPS", 4);
    unsigned steps_per = EnvU("GSBENCH_STEPS_PER", 8);
    std::string bdev = EnvS("GSBENCH_BDEV", "ram");
    unsigned cap_mb  = EnvU("GSBENCH_BDEV_CAP_MB", 512);
    std::string bdev_path = EnvS("GSBENCH_BDEV_PATH", "./gsbench_bdev.dat");
    std::string disk_dir  = EnvS("GSBENCH_DISK_DIR", "./gsbench_raw_out");
    // Raw arm O_DIRECT (cache-bypass) for the disk comparison; set 0 for buffered writes
    // when the raw target is a RAM tier (tmpfs, which rejects O_DIRECT) — the fair
    // software-path comparison vs CLIO's kRam bdev.
    unsigned raw_odirect  = EnvU("GSBENCH_RAW_ODIRECT", 1);
    // Raw arm fsync per snapshot: DURABILITY PARITY with CLIO (whose bdev waits for each
    // write to commit). Without it, a buffered raw write just lands in RAM/page-cache and
    // isn't really persisted — doing less work than CLIO. Default on.
    unsigned raw_fsync    = EnvU("GSBENCH_RAW_FSYNC", 1);

    unsigned steps() const { return snaps * steps_per; }
    uint64_t cells() const { return uint64_t(N) * N; }
    uint64_t grid_bytes() const { return cells() * sizeof(float); }
    // total bytes persisted by the whole run (one v-grid per snapshot).
    uint64_t total_bytes() const { return grid_bytes() * snaps; }
};

// ---- shared sim scaffolding ------------------------------------------------

struct Grids { float *u_curr, *u_next, *v_curr, *v_next; };
Grids MakeGrids(unsigned N) {
    uint64_t cells = uint64_t(N) * N, bytes = cells * sizeof(float);
    Grids g{};
    REQUIRE(cudaMalloc(&g.u_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.u_next, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_curr, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&g.v_next, bytes) == cudaSuccess);
    std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
    unsigned lo = N/2 - 3, hi = N/2 + 3;
    for (unsigned y = lo; y < hi; ++y)
        for (unsigned x = lo; x < hi; ++x) v0[uint64_t(y)*N + x] = 1.0f;
    ctp::GpuApi::Memcpy(g.u_curr, u0.data(), bytes);
    ctp::GpuApi::Memcpy(g.v_curr, v0.data(), bytes);
    return g;
}
void FreeGrids(Grids& g) {
    cudaFree(g.u_curr); cudaFree(g.u_next); cudaFree(g.v_curr); cudaFree(g.v_next);
}

// FNV-1a over a host byte buffer (cross-arm "identical computation" proof).
uint64_t Fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// The ONE timed sim loop shared by every arm. `snap(si, v_curr)` persists snapshot si;
// `finalize()` runs inside the timed region (async drain). No per-step Synchronize so
// the stream can overlap; one Synchronize closes the region. Returns wall-clock ms and
// accumulates a checksum of every snapshot's v-grid (host-read) for cross-arm equality.
template <class SnapFn, class FinalizeFn>
double RunSim(const Cfg& cfg, Grids& g, SnapFn snap, FinalizeFn finalize,
              uint64_t* checksum_out) {
    unsigned N = cfg.N;
    uint64_t cells = cfg.cells();
    unsigned threads = 256, blocks = unsigned((cells + threads - 1) / threads);
    ctp::GpuApi::Synchronize();  // settle the seed before timing
    auto t0 = std::chrono::steady_clock::now();
    unsigned si = 0;
    for (unsigned step = 1; step <= cfg.steps(); ++step) {
        GsStepKernel<<<blocks, threads>>>(g.u_curr, g.v_curr, g.u_next, g.v_next,
                                          kGs, N);
        std::swap(g.u_curr, g.u_next);
        std::swap(g.v_curr, g.v_next);
        if (step % cfg.steps_per != 0) continue;
        snap(si, g.v_curr);
        ++si;
    }
    finalize();
    ctp::GpuApi::Synchronize();
    auto t1 = std::chrono::steady_clock::now();
    (void)checksum_out;  // checksum computed by arms post-run from persisted bytes
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintResult(const char* arm, const Cfg& cfg, double ms, uint64_t checksum) {
    double mb = double(cfg.total_bytes()) / (1024.0 * 1024.0);
    std::fprintf(stderr,
        "GSBENCH_RESULT arm=%s N=%u chunks=%u snaps=%u steps=%u bdev=%s "
        "MB=%.1f ms=%.2f MBps=%.1f checksum=%llu\n",
        arm, cfg.N, cfg.chunks, cfg.snaps, cfg.steps(), cfg.bdev.c_str(),
        mb, ms, mb / (ms / 1000.0), (unsigned long long)checksum);
}

// ---- CLIO env bring-up (configurable bdev) ---------------------------------

struct BenchEnv {
    clio::cte::core::TagId probe_tag;
    BenchEnv(const Cfg& cfg) {
        using namespace std::chrono_literals;
        namespace bdev = clio::run::bdev;
        std::fprintf(stderr, "[bench] bringing up server (bdev=%s cap=%uMB)\n",
                     cfg.bdev.c_str(), cfg.cap_mb);
        if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer))
            throw std::runtime_error("CHIMAERA_INIT(kServer) failed");
        if (!clio::cte::core::CLIO_CTE_CLIENT_INIT())
            throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
        auto* cte = CLIO_CTE_CLIENT;
        cte->Init(clio::cte::core::kCtePoolId);
        clio::cte::core::CreateParams params;
        auto ct = cte->AsyncCreate(chi::PoolQuery::Dynamic(),
                                   clio::cte::core::kCtePoolName,
                                   clio::cte::core::kCtePoolId, params);
        ct.Wait();
        if (ct->GetReturnCode() != 0) throw std::runtime_error("CTE create failed");
        std::this_thread::sleep_for(50ms);

        const chi::u64 cap = chi::u64(cfg.cap_mb) << 20;
        const bool is_file = (cfg.bdev == "file");
        const bdev::BdevType type = is_file ? bdev::BdevType::kFile
                                            : bdev::BdevType::kRam;
        // For kFile the bdev name IS the on-disk file path (O_DIRECT). For kRam it is
        // just an identifier.
        const std::string name = is_file ? cfg.bdev_path : std::string("gsbench_ram");
        chi::PoolId bdev_pool_id(960, 0);
        bdev::Client bclient(bdev_pool_id);
        auto bc = bclient.AsyncCreate(chi::PoolQuery::Dynamic(), name, bdev_pool_id,
                                      type, cap);
        bc.Wait();
        if (bc->GetReturnCode() != 0) throw std::runtime_error("bdev create failed");
        std::this_thread::sleep_for(50ms);
        auto rt = cte->AsyncRegisterTarget(name, type, cap, chi::PoolQuery::Local(),
                                           bdev_pool_id);
        rt.Wait();
        if (rt->GetReturnCode() != 0) throw std::runtime_error("RegisterTarget failed");
        std::this_thread::sleep_for(50ms);
        std::fprintf(stderr, "[bench] server ready\n");
    }
};

clio::cte::core::TagId MakeTag(const char* name) {
    auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    return t->tag_id_;
}

std::vector<byte_t> HostReadBlob(clio::cte::core::TagId tag, const std::string& name,
                                 uint64_t size) {
    ctp::ipc::FullPtr<char> buf = CLIO_CPU_IPC->AllocateBuffer(size);
    REQUIRE(!buf.IsNull());
    std::memset(buf.ptr_, 0, size);
    ctp::ipc::ShmPtr<> shm = buf.shm_.template Cast<void>();
    auto t = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, chi::u64(0), size,
                                           chi::u32(0), shm);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    std::vector<byte_t> out(size);
    std::memcpy(out.data(), buf.ptr_, size);
    return out;
}

// Pre-create `snaps` snapshot datasets (distinct path => distinct tag). Kept well under
// the ~16-large-backend ceiling by the caller (snaps <= ~12).
void MakeSnapDatasets(chi::IpcManager* ipc, chi::IpcManagerGpuInfo gpu_info,
                      const char* prefix, const Cfg& cfg,
                      std::vector<kvhdf5::GpuCteDataset>& out,
                      std::vector<clio::cte::core::TagId>& tags) {
    kvhdf5::Layout layout{/*dims=*/{cfg.cells()},
                          /*chunk_dims=*/{cfg.cells() / cfg.chunks},
                          /*elem_size=*/sizeof(float)};
    REQUIRE(layout.ChunkCount() == cfg.chunks);
    out.clear(); tags.clear(); out.reserve(cfg.snaps);
    for (unsigned s = 0; s < cfg.snaps; ++s) {
        char path[160];
        std::snprintf(path, sizeof(path), "%s/v/step_%04u", prefix, s);
        out.emplace_back(kvhdf5::GpuCteDataset::FromPath(
            ipc, gpu_info, /*gpu_id=*/0, CLIO_CTE_CLIENT, path, layout));
        tags.push_back(MakeTag(kvhdf5::tagpath::CanonicalTag(path).c_str()));
    }
}

// Read every snapshot back and fold into one checksum (proves what was persisted).
uint64_t ChecksumSnapshots(const std::vector<clio::cte::core::TagId>& tags,
                           const Cfg& cfg) {
    const uint64_t chunk_bytes = cfg.grid_bytes() / cfg.chunks;
    uint64_t h = 1469598103934665603ull;
    for (unsigned s = 0; s < cfg.snaps; ++s)
        for (unsigned c = 0; c < cfg.chunks; ++c) {
            auto got = HostReadBlob(tags[s], std::to_string(c), chunk_bytes);
            h = Fnv1a(got.data(), got.size(), h);
        }
    return h;
}

// Run a CLIO arm (async=false -> sync). Returns ms; fills checksum from persisted bytes.
double RunClioArm(const Cfg& cfg, bool async, const char* prefix, uint64_t* checksum) {
    auto* ipc = CLIO_CPU_IPC;
    REQUIRE(ipc->GetGpuIpcManager() != nullptr);
    chi::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
    REQUIRE(gpu_info.gpu2cpu_queue != nullptr);

    std::vector<kvhdf5::GpuCteDataset> ds;
    std::vector<clio::cte::core::TagId> tags;
    MakeSnapDatasets(ipc, gpu_info, prefix, cfg, ds, tags);

    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        if (async) TwSnapFireKernel<<<1, 256>>>(ds[si].Handle(), v_curr);
        else       TwSnapSyncKernel<<<1, 256>>>(ds[si].Handle(), v_curr);
    };
    auto finalize = [&]() {
        if (async) for (auto& d : ds) TwDrainKernel<<<1, 32>>>(d.Handle());
    };
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);
    *checksum = ChecksumSnapshots(tags, cfg);
    return ms;
}

// ---- raw (no-CLIO) disk arm -----------------------------------------------

void EnsureDir(const std::string& d) {
    if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("mkdir " + d);
}
// Write in <=1 MiB O_DIRECT chunks (matches the CLIO bdev's block loop; a single huge
// O_DIRECT pwrite from CUDA-pinned memory hit a ~5x-slow kernel path on this box).
void WriteAllAt(int fd, off_t off, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    constexpr size_t kBlk = 1u << 20;
    while (bytes) {
        size_t want = bytes < kBlk ? bytes : kBlk;
        ssize_t n = pwrite(fd, p, want, off);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("pwrite"); }
        p += n; off += n; bytes -= size_t(n);
    }
}

// A competent (deliberately NOT maximally-tuned) decoupled writer: ONE background thread
// + a small pinned-buffer pool (standard double-buffered checkpoint I/O). It lets disk
// I/O OVERLAP the subsequent sim steps instead of stalling inline — the same structural
// benefit CLIO gets by offloading I/O to its server. No libaio / queue-depth / thread
// fan-out (that would be an "expert" baseline; we keep it fair). O_DIRECT for cache-bypass
// parity with CLIO's kFile bdev.
class BgWriter {
public:
    BgWriter(std::string path, uint64_t gbytes, uint64_t wbytes, unsigned nbuf,
             unsigned snaps, bool odirect, bool do_fsync)
        : gbytes_(gbytes), wbytes_(wbytes), fsync_(do_fsync) {
        // ONE pre-allocated checkpoint file, snapshots written at distinct offsets — same
        // as CLIO's bdev (single truncated file, offset per blob). Avoids the per-snapshot
        // file create/O_TRUNC + async-discard churn that throttled the 12-fresh-files
        // pattern ~5x when the writes were spaced out by GPU work.
        int flags = O_WRONLY | O_CREAT | O_TRUNC | (odirect ? O_DIRECT : 0);
        fd_ = open(path.c_str(), flags, 0644);
        REQUIRE(fd_ >= 0);
        REQUIRE(ftruncate(fd_, off_t(snaps) * off_t(wbytes_)) == 0);  // preallocate size
        bufs_.resize(nbuf);
        for (unsigned i = 0; i < nbuf; ++i) {
            REQUIRE(cudaMallocHost(reinterpret_cast<void**>(&bufs_[i]), wbytes_)
                    == cudaSuccess);                 // pinned: fast D2H
            std::memset(bufs_[i] + gbytes_, 0, wbytes_ - gbytes_);  // O_DIRECT pad tail
            free_.push_back(int(i));
        }
        th_ = std::thread([this] { Run(); });
    }
    // Producer (CUDA thread): grab a free buffer to D2H into (blocks if all in flight).
    uint8_t* Acquire(int* idx) {
        std::unique_lock<std::mutex> lk(m_);
        cv_free_.wait(lk, [this] { return !free_.empty(); });
        int i = free_.back(); free_.pop_back();
        *idx = i; return bufs_[i];
    }
    // Producer: buffer filled (D2H complete) -> hand to the writer.
    void Submit(int idx, unsigned s) {
        { std::lock_guard<std::mutex> lk(m_); work_.push_back({idx, s}); }
        cv_work_.notify_one();
    }
    // Drain + join (call INSIDE the timed region). Yields writer busy-ms. Checksum is
    // computed by the caller AFTER timing (readback), matching the CLIO arms.
    void Finish(double* writer_ms) {
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_work_.notify_one();
        th_.join();
        if (fd_ >= 0) close(fd_);
        for (auto* b : bufs_) cudaFreeHost(b);
        *writer_ms = writer_ms_;
        REQUIRE(!err_);
    }
private:
    void Run() {
        for (;;) {
            std::pair<int, unsigned> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [this] { return !work_.empty() || done_; });
                if (work_.empty() && done_) return;
                job = work_.front(); work_.pop_front();
            }
            auto t0 = std::chrono::steady_clock::now();
            uint8_t* buf = bufs_[job.first];
            // NB: checksum is computed AFTER timing (post-run readback, like the CLIO arms)
            // — folding a scalar FNV over 1.9 GB here would add ~6 s to the timed region and
            // unfairly penalize raw vs CLIO (whose ChecksumSnapshots runs post-timing).
            WriteAllAt(fd_, off_t(job.second) * off_t(wbytes_), buf, wbytes_);  // offset slot
            if (fsync_ && fdatasync(fd_) != 0) err_ = true;  // durability parity with CLIO
            writer_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            { std::lock_guard<std::mutex> lk(m_); free_.push_back(job.first); }
            cv_free_.notify_one();
        }
    }
    int fd_ = -1; uint64_t gbytes_, wbytes_; bool fsync_ = true;
    std::vector<uint8_t*> bufs_;
    std::mutex m_; std::condition_variable cv_free_, cv_work_;
    std::vector<int> free_; std::deque<std::pair<int, unsigned>> work_;
    bool done_ = false, err_ = false; std::thread th_;
    double writer_ms_ = 0;
};

// Raw arm (no CLIO): identical sim + timed loop; each snapshot D2Hs the v-grid into a
// pooled pinned buffer and hands it to a background writer thread (O_DIRECT, cache-bypass
// parity). The writer overlaps the next sim steps; the final drain is inside the timed
// region. This is the FAIR baseline — a naive INLINE write here is throttled ~5x by
// interleaving with GPU compute/D2H (measured), which is a harness artifact, not real cost.
double RunRawArm(const Cfg& cfg, uint64_t* checksum) {
    EnsureDir(cfg.disk_dir);
    const uint64_t gbytes = cfg.grid_bytes();
    constexpr uint64_t kAlign = 4096;
    const uint64_t wbytes = (gbytes + kAlign - 1) & ~(kAlign - 1);  // O_DIRECT length
    BgWriter writer(cfg.disk_dir + "/checkpoint.bin", gbytes, wbytes, /*nbuf=*/3,
                    cfg.snaps, /*odirect=*/cfg.raw_odirect != 0,
                    /*do_fsync=*/cfg.raw_fsync != 0);  // one pre-allocated file, durable

    Grids g = MakeGrids(cfg.N);
    auto snap = [&](unsigned si, float* v_curr) {
        int idx;
        uint8_t* buf = writer.Acquire(&idx);
        // SYNCHRONOUS D2H on the DEFAULT stream (ordered after the step that produced
        // v_curr; a side stream would race the compute -> checksum divergence). Then hand
        // the buffer to the writer, which writes it while the NEXT sim steps run.
        cudaMemcpy(buf, v_curr, gbytes, cudaMemcpyDeviceToHost);
        writer.Submit(idx, si);
    };
    double writer_ms = 0;
    auto finalize = [&]() { writer.Finish(&writer_ms); };  // drain inside timed region
    double ms = RunSim(cfg, g, snap, finalize, nullptr);
    FreeGrids(g);

    // Checksum AFTER timing: read the persisted checkpoint back and fold FNV in snapshot
    // order — mirrors the CLIO arms' post-timing ChecksumSnapshots (scalar FNV over 1.9 GB
    // is ~6 s and must NOT be inside either arm's timed region).
    uint64_t h = 1469598103934665603ull;
    {
        int fd = open((cfg.disk_dir + "/checkpoint.bin").c_str(), O_RDONLY);
        REQUIRE(fd >= 0);
        std::vector<uint8_t> rb(gbytes);
        for (unsigned s = 0; s < cfg.snaps; ++s) {
            off_t base = off_t(s) * off_t(wbytes);
            size_t got = 0;
            while (got < gbytes) {
                ssize_t n = pread(fd, rb.data() + got, gbytes - got, base + off_t(got));
                REQUIRE(n > 0);
                got += size_t(n);
            }
            h = Fnv1a(rb.data(), gbytes, h);
        }
        close(fd);
    }
    std::fprintf(stderr,
        "[raw] total=%.1f ms  writer-busy=%.1f ms (overlapped with compute)  nbuf=3\n",
        ms, writer_ms);
    *checksum = h;
    return ms;
}

}  // namespace

// ---- the three arm TEST_CASEs (run each in its OWN process) ----------------

// raw arm needs no CLIO server.
TEST_CASE("GSBENCH raw disk (no CLIO)", "[.][integration][gpu][gsbench][gsbench_raw]") {
    Cfg cfg;
    uint64_t checksum = 0;
    double ms = RunRawArm(cfg, &checksum);
    PrintResult("raw", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

TEST_CASE("GSBENCH clio sync", "[.][integration][gpu][gsbench][gsbench_sync]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, /*async=*/false, "results/gsbench/sync", &checksum);
    PrintResult("sync", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

TEST_CASE("GSBENCH clio async", "[.][integration][gpu][gsbench][gsbench_async]") {
    Cfg cfg;
    static BenchEnv env(cfg);
    uint64_t checksum = 0;
    double ms = RunClioArm(cfg, /*async=*/true, "results/gsbench/async", &checksum);
    PrintResult("async", cfg, ms, checksum);
    REQUIRE(ms > 0.0);
}

#endif  // !CTP_IS_DEVICE_PASS

#else
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

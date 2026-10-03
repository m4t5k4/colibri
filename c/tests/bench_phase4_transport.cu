/* Phase 4A0 transport/topology probe for Colibri GLM-5.3.
 *
 * Measures the machine, not model math:
 *   - CUDA device inventory and PCI identity
 *   - pinned H2D and D2H bandwidth for every visible GPU
 *   - peer-access matrix and directional cudaMemcpyPeerAsync bandwidth
 *   - Linux io_uring + O_DIRECT throughput into CUDA-registered 4K-aligned buffers
 *
 * Build from c/:
 *   make phase4-transport-bench CUDA=1 CUDA_ARCH=sm_86
 *
 * Example on the target rig:
 *   ./phase4_transport_bench --file /srv/models-fast/colibri/glm53-flash-i4/experts.bin \
 *       --read-mib 16 --qd 8 --batches 64 --copy-mib 64 --copy-iters 32
 *
 * stdout is one JSON document. Diagnostics go to stderr.
 */

#include <cuda_runtime.h>

#if !defined(__linux__)
#error "phase4 transport probe currently requires Linux (io_uring + O_DIRECT)"
#endif

#include "../uring.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kDirectAlignment = 4096;

struct Options {
    std::string file;
    size_t read_bytes = 16ull << 20;
    unsigned qd = 8;
    unsigned batches = 32;
    unsigned io_workers = 2;
    size_t copy_bytes = 64ull << 20;
    unsigned copy_iters = 16;
    size_t peer_bytes = 32ull << 20;
    unsigned peer_iters = 8;
};

static double now_s() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static size_t align_down(size_t value, size_t alignment) {
    return value - value % alignment;
}

static size_t align_up(size_t value, size_t alignment) {
    const size_t rem = value % alignment;
    return rem ? value + alignment - rem : value;
}

[[noreturn]] static void die(const char *what) {
    std::fprintf(stderr, "phase4-transport: %s: %s\n", what, std::strerror(errno));
    std::exit(2);
}

[[noreturn]] static void cuda_die(const char *what, cudaError_t err) {
    std::fprintf(stderr, "phase4-transport: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(2);
}

static void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) cuda_die(what, err);
}

static unsigned parse_u(const char *text, const char *name) {
    char *end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno || !end || *end || value == 0 || value > 1u << 20) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return static_cast<unsigned>(value);
}

static size_t parse_mib(const char *text, const char *name) {
    const unsigned value = parse_u(text, name);
    if (value > (SIZE_MAX >> 20)) {
        std::fprintf(stderr, "%s too large: %s\n", name, text);
        std::exit(2);
    }
    return static_cast<size_t>(value) << 20;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s [options]\n"
        "  --file PATH       regular file used for O_DIRECT io_uring reads\n"
        "  --read-mib N      bytes per NVMe read (default 16 MiB)\n"
        "  --qd N            io_uring queue depth (default 8)\n"
        "  --batches N       qd-sized read batches (default 32)\n"
        "  --io-workers N    max io-wq workers (default 2)\n"
        "  --copy-mib N      H2D/D2H copy size (default 64 MiB)\n"
        "  --copy-iters N    H2D/D2H repetitions (default 16)\n"
        "  --peer-mib N      P2P copy size (default 32 MiB)\n"
        "  --peer-iters N    P2P repetitions (default 8)\n",
        argv0);
}

static Options parse_options(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--file")) o.file = need("--file");
        else if (!std::strcmp(argv[i], "--read-mib")) o.read_bytes = parse_mib(need("--read-mib"), "--read-mib");
        else if (!std::strcmp(argv[i], "--qd")) o.qd = parse_u(need("--qd"), "--qd");
        else if (!std::strcmp(argv[i], "--batches")) o.batches = parse_u(need("--batches"), "--batches");
        else if (!std::strcmp(argv[i], "--io-workers")) o.io_workers = parse_u(need("--io-workers"), "--io-workers");
        else if (!std::strcmp(argv[i], "--copy-mib")) o.copy_bytes = parse_mib(need("--copy-mib"), "--copy-mib");
        else if (!std::strcmp(argv[i], "--copy-iters")) o.copy_iters = parse_u(need("--copy-iters"), "--copy-iters");
        else if (!std::strcmp(argv[i], "--peer-mib")) o.peer_bytes = parse_mib(need("--peer-mib"), "--peer-mib");
        else if (!std::strcmp(argv[i], "--peer-iters")) o.peer_iters = parse_u(need("--peer-iters"), "--peer-iters");
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            std::exit(2);
        }
    }
    o.read_bytes = align_up(o.read_bytes, kDirectAlignment);
    return o;
}

struct RegisteredBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    RegisteredBuffer() = default;
    explicit RegisteredBuffer(size_t n) { allocate(n); }
    RegisteredBuffer(const RegisteredBuffer &) = delete;
    RegisteredBuffer &operator=(const RegisteredBuffer &) = delete;
    RegisteredBuffer(RegisteredBuffer &&other) noexcept {
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    RegisteredBuffer &operator=(RegisteredBuffer &&other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
    ~RegisteredBuffer() { release(); }

    void allocate(size_t n) {
        release();
        void *p = nullptr;
        const int rc = posix_memalign(&p, kDirectAlignment, n);
        if (rc != 0) {
            errno = rc;
            die("posix_memalign");
        }
        std::memset(p, 0xa5, n);
        cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            std::free(p);
            cuda_die("cudaHostRegister", err);
        }
        ptr = p;
        bytes = n;
    }

    void release() {
        if (ptr) {
            (void)cudaHostUnregister(ptr);
            std::free(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

struct DeviceCopyResult {
    double h2d_gib_s = 0.0;
    double d2h_gib_s = 0.0;
};

static double gib_per_s(size_t bytes, unsigned iterations, double seconds) {
    const double gib = static_cast<double>(bytes) * iterations / static_cast<double>(1ull << 30);
    return seconds > 0.0 ? gib / seconds : 0.0;
}

static DeviceCopyResult bench_host_device(int device, void *host, size_t bytes, unsigned iterations) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice host-device bench");
    void *gpu = nullptr;
    cudaStream_t stream = nullptr;
    cuda_check(cudaMalloc(&gpu, bytes), "cudaMalloc host-device bench");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate host-device bench");

    cuda_check(cudaMemcpyAsync(gpu, host, bytes, cudaMemcpyHostToDevice, stream), "H2D warmup");
    cuda_check(cudaStreamSynchronize(stream), "H2D warmup sync");

    double start = now_s();
    for (unsigned i = 0; i < iterations; ++i)
        cuda_check(cudaMemcpyAsync(gpu, host, bytes, cudaMemcpyHostToDevice, stream), "H2D copy");
    cuda_check(cudaStreamSynchronize(stream), "H2D sync");
    double h2d = now_s() - start;

    cuda_check(cudaMemcpyAsync(host, gpu, bytes, cudaMemcpyDeviceToHost, stream), "D2H warmup");
    cuda_check(cudaStreamSynchronize(stream), "D2H warmup sync");

    start = now_s();
    for (unsigned i = 0; i < iterations; ++i)
        cuda_check(cudaMemcpyAsync(host, gpu, bytes, cudaMemcpyDeviceToHost, stream), "D2H copy");
    cuda_check(cudaStreamSynchronize(stream), "D2H sync");
    double d2h = now_s() - start;

    (void)cudaStreamDestroy(stream);
    (void)cudaFree(gpu);
    return {gib_per_s(bytes, iterations, h2d), gib_per_s(bytes, iterations, d2h)};
}

static double bench_peer(int src_device, int dst_device, size_t bytes, unsigned iterations) {
    int can = 0;
    cuda_check(cudaDeviceCanAccessPeer(&can, dst_device, src_device), "cudaDeviceCanAccessPeer");
    if (!can) return 0.0;

    void *src = nullptr;
    void *dst = nullptr;
    cudaStream_t stream = nullptr;

    cuda_check(cudaSetDevice(src_device), "cudaSetDevice peer src");
    cuda_check(cudaMalloc(&src, bytes), "cudaMalloc peer src");
    cuda_check(cudaMemset(src, 0x5a, bytes), "cudaMemset peer src");

    cuda_check(cudaSetDevice(dst_device), "cudaSetDevice peer dst");
    cudaError_t peer_err = cudaDeviceEnablePeerAccess(src_device, 0);
    if (peer_err != cudaSuccess && peer_err != cudaErrorPeerAccessAlreadyEnabled)
        cuda_die("cudaDeviceEnablePeerAccess", peer_err);
    if (peer_err == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
    cuda_check(cudaMalloc(&dst, bytes), "cudaMalloc peer dst");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate peer");

    cuda_check(cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream), "P2P warmup");
    cuda_check(cudaStreamSynchronize(stream), "P2P warmup sync");

    const double start = now_s();
    for (unsigned i = 0; i < iterations; ++i)
        cuda_check(cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream), "P2P copy");
    cuda_check(cudaStreamSynchronize(stream), "P2P sync");
    const double elapsed = now_s() - start;

    (void)cudaStreamDestroy(stream);
    (void)cudaFree(dst);
    cuda_check(cudaSetDevice(src_device), "cudaSetDevice peer cleanup src");
    (void)cudaFree(src);
    return gib_per_s(bytes, iterations, elapsed);
}

struct IoResult {
    bool ran = false;
    uint64_t bytes = 0;
    double seconds = 0.0;
    double gib_s = 0.0;
};

static IoResult bench_io_uring(const Options &o, std::vector<RegisteredBuffer> &buffers) {
    IoResult result;
    if (o.file.empty()) return result;

    const int fd = open(o.file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) die("open O_DIRECT input");

    struct stat st{};
    if (fstat(fd, &st) != 0) {
        close(fd);
        die("fstat input");
    }
    if (!S_ISREG(st.st_mode)) {
        std::fprintf(stderr, "phase4-transport: --file must name a regular file\n");
        close(fd);
        std::exit(2);
    }

    const uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
    if (file_bytes < o.read_bytes) {
        std::fprintf(stderr, "phase4-transport: file is smaller than one read (%" PRIu64 " < %zu)\n",
                     file_bytes, o.read_bytes);
        close(fd);
        std::exit(2);
    }
    const uint64_t max_offset = align_down(static_cast<size_t>(file_bytes - o.read_bytes), kDirectAlignment);

    ColiUring ring{};
    const unsigned ring_entries = std::max(2u, o.qd * 2u);
    if (coli_uring_init(&ring, ring_entries) != 0) {
        close(fd);
        die("io_uring_setup");
    }
    if (coli_uring_set_workers(&ring, o.io_workers) != 0) {
        std::fprintf(stderr, "phase4-transport: warning: IORING_REGISTER_IOWQ_MAX_WORKERS failed: %s\n",
                     std::strerror(errno));
    }

    buffers.clear();
    buffers.reserve(o.qd);
    for (unsigned i = 0; i < o.qd; ++i) buffers.emplace_back(o.read_bytes);

    uint64_t completed_bytes = 0;
    const double start = now_s();
    for (unsigned batch = 0; batch < o.batches; ++batch) {
        for (unsigned i = 0; i < o.qd; ++i) {
            const uint64_t ordinal = static_cast<uint64_t>(batch) * o.qd + i;
            uint64_t offset = 0;
            if (max_offset) {
                offset = (ordinal * o.read_bytes) % (max_offset + kDirectAlignment);
                offset = align_down(static_cast<size_t>(offset), kDirectAlignment);
            }
            if (coli_uring_prep_read(&ring, fd, buffers[i].ptr, o.read_bytes,
                                     static_cast<int64_t>(offset), i) != 0) {
                coli_uring_close(&ring);
                close(fd);
                die("io_uring prep read");
            }
        }
        if (coli_uring_enter(&ring, 1) < 0) {
            coli_uring_close(&ring);
            close(fd);
            die("io_uring enter");
        }

        unsigned complete = 0;
        while (complete < o.qd) {
            io_uring_cqe cqe{};
            bool reaped = false;
            while (coli_uring_peek(&ring, &cqe)) {
                reaped = true;
                if (cqe.res < 0) {
                    errno = -cqe.res;
                    coli_uring_close(&ring);
                    close(fd);
                    die("io_uring read completion");
                }
                if (static_cast<size_t>(cqe.res) != o.read_bytes) {
                    std::fprintf(stderr, "phase4-transport: short direct read: %d of %zu bytes\n",
                                 cqe.res, o.read_bytes);
                    coli_uring_close(&ring);
                    close(fd);
                    std::exit(2);
                }
                completed_bytes += static_cast<uint64_t>(cqe.res);
                ++complete;
            }
            if (complete < o.qd && !reaped && coli_uring_enter(&ring, 1) < 0) {
                coli_uring_close(&ring);
                close(fd);
                die("io_uring wait");
            }
        }
    }
    const double elapsed = now_s() - start;

    coli_uring_close(&ring);
    close(fd);
    result.ran = true;
    result.bytes = completed_bytes;
    result.seconds = elapsed;
    result.gib_s = elapsed > 0.0
        ? static_cast<double>(completed_bytes) / static_cast<double>(1ull << 30) / elapsed
        : 0.0;
    return result;
}

static void json_string(const char *s) {
    std::putchar('"');
    for (; *s; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        switch (c) {
        case '\\': std::fputs("\\\\", stdout); break;
        case '"': std::fputs("\\\"", stdout); break;
        case '\n': std::fputs("\\n", stdout); break;
        case '\r': std::fputs("\\r", stdout); break;
        case '\t': std::fputs("\\t", stdout); break;
        default:
            if (c < 0x20) std::printf("\\u%04x", c);
            else std::putchar(c);
        }
    }
    std::putchar('"');
}

}  // namespace

int main(int argc, char **argv) {
    const Options o = parse_options(argc, argv);

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        std::fprintf(stderr, "phase4-transport: no CUDA devices visible\n");
        return 2;
    }

    RegisteredBuffer copy_host(o.copy_bytes);
    std::vector<DeviceCopyResult> copies(static_cast<size_t>(device_count));
    std::vector<cudaDeviceProp> props(static_cast<size_t>(device_count));
    for (int d = 0; d < device_count; ++d) {
        cuda_check(cudaGetDeviceProperties(&props[d], d), "cudaGetDeviceProperties");
        copies[d] = bench_host_device(d, copy_host.ptr, o.copy_bytes, o.copy_iters);
    }

    std::vector<int> peer_access(static_cast<size_t>(device_count) * device_count, 0);
    std::vector<double> peer_gib_s(static_cast<size_t>(device_count) * device_count, 0.0);
    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) continue;
            int can = 0;
            cuda_check(cudaDeviceCanAccessPeer(&can, dst, src), "cudaDeviceCanAccessPeer matrix");
            peer_access[static_cast<size_t>(src) * device_count + dst] = can;
            if (can) {
                peer_gib_s[static_cast<size_t>(src) * device_count + dst] =
                    bench_peer(src, dst, o.peer_bytes, o.peer_iters);
            }
        }
    }

    std::vector<RegisteredBuffer> io_buffers;
    const IoResult io = bench_io_uring(o, io_buffers);

    std::printf("{\n  \"schema\": \"colibri-phase4-transport-v1\",\n");
    std::printf("  \"device_count\": %d,\n", device_count);
    std::printf("  \"copy_bytes\": %zu,\n  \"copy_iters\": %u,\n", o.copy_bytes, o.copy_iters);
    std::printf("  \"devices\": [\n");
    for (int d = 0; d < device_count; ++d) {
        const auto &p = props[d];
        std::printf("    {\"id\":%d,\"name\":", d);
        json_string(p.name);
        std::printf(",\"cc\":\"%d.%d\",\"pci_domain\":%d,\"pci_bus\":%d,\"pci_device\":%d,"
                    "\"total_global_mem\":%zu,\"h2d_gib_s\":%.6f,\"d2h_gib_s\":%.6f}%s\n",
                    p.major, p.minor, p.pciDomainID, p.pciBusID, p.pciDeviceID,
                    static_cast<size_t>(p.totalGlobalMem), copies[d].h2d_gib_s, copies[d].d2h_gib_s,
                    d + 1 == device_count ? "" : ",");
    }
    std::printf("  ],\n");

    std::printf("  \"peer\": [\n");
    bool first_peer = true;
    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) continue;
            if (!first_peer) std::printf(",\n");
            first_peer = false;
            const size_t idx = static_cast<size_t>(src) * device_count + dst;
            std::printf("    {\"src\":%d,\"dst\":%d,\"access\":%s,\"gib_s\":%.6f}",
                        src, dst, peer_access[idx] ? "true" : "false", peer_gib_s[idx]);
        }
    }
    std::printf("\n  ],\n");

    if (io.ran) {
        std::printf("  \"nvme\": {\"file\":");
        json_string(o.file.c_str());
        std::printf(",\"direct\":true,\"read_bytes\":%zu,\"qd\":%u,\"batches\":%u,"
                    "\"io_workers\":%u,\"bytes\":%" PRIu64 ",\"seconds\":%.6f,\"gib_s\":%.6f}\n",
                    o.read_bytes, o.qd, o.batches, o.io_workers, io.bytes, io.seconds, io.gib_s);
    } else {
        std::printf("  \"nvme\": null\n");
    }
    std::printf("}\n");
    return 0;
}

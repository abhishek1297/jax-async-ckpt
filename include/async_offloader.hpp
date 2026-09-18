#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

class AsyncOffloader {
   private:
    static constexpr size_t SECTOR_SIZE = 4096;

    size_t single_buffer_capacity_ = 0;
    void* host_buffers_[2] = {nullptr, nullptr};
    int active_buffer_idx_ = 0;

    cudaStream_t copy_stream_ = nullptr;
    struct io_uring ring_;
    bool ring_initialized_ = false;

    std::vector<int> active_fds_;
    size_t pending_writes_ = 0;
    size_t pending_reads_ = 0;

    size_t align_to_sector(size_t bytes) const {
        return (bytes + SECTOR_SIZE - 1) & ~(SECTOR_SIZE - 1);
    }

   public:
    AsyncOffloader(size_t max_buffer_bytes, unsigned io_ring_entries = 64);
    ~AsyncOffloader();

    int swap_buffers() {
        active_buffer_idx_ = 1 - active_buffer_idx_;
        return active_buffer_idx_;
    }

    // Save Pipeline Methods
    void offload_gpu_to_host_async(uintptr_t gpu_ptr_address, size_t bytes, size_t host_offset);
    void write_host_to_nvme_async(const std::string& filepath, size_t bytes, size_t host_offset);

    // Restore Pipeline Methods
    void read_nvme_to_host_async(const std::string& filepath, size_t bytes, size_t host_offset);
    void load_host_to_gpu_async(uintptr_t gpu_ptr_address, size_t bytes, size_t host_offset);

    void wait_gpu_to_host_completion();
    void flush_nvme_writes();
};
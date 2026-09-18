#include "async_offloader.hpp"

#include <iostream>

#define CUDA_CHECK(cmd)                                                                    \
    do {                                                                                   \
        cudaError_t e = cmd;                                                               \
        if (e != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(e)); \
        }                                                                                  \
    } while (0)

AsyncOffloader::AsyncOffloader(size_t max_buffer_bytes, unsigned io_ring_entries)
    : single_buffer_capacity_(align_to_sector(max_buffer_bytes)) {
    CUDA_CHECK(cudaHostAlloc(&host_buffers_[0], single_buffer_capacity_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host_buffers_[1], single_buffer_capacity_, cudaHostAllocDefault));

    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

    int ret = io_uring_queue_init(io_ring_entries, &ring_, 0);
    if (ret < 0) {
        cudaFreeHost(host_buffers_[0]);
        cudaFreeHost(host_buffers_[1]);
        throw std::runtime_error("Failed to initialize io_uring queue: " + std::to_string(-ret));
    }
    ring_initialized_ = true;
}

AsyncOffloader::~AsyncOffloader() {
    flush_nvme_writes();
    if (copy_stream_)
        cudaStreamDestroy(copy_stream_);
    if (host_buffers_[0])
        cudaFreeHost(host_buffers_[0]);
    if (host_buffers_[1])
        cudaFreeHost(host_buffers_[1]);
    if (ring_initialized_)
        io_uring_queue_exit(&ring_);
}

void AsyncOffloader::offload_gpu_to_host_async(uintptr_t gpu_ptr_address, size_t bytes,
                                               size_t host_offset) {
    if (host_offset + bytes > single_buffer_capacity_) {
        throw std::invalid_argument("Transfer size exceeds allocated single buffer capacity.");
    }

    const void* gpu_ptr = reinterpret_cast<const void*>(gpu_ptr_address);
    // Offset target memory address by host_offset
    uint8_t* target_host_buffer =
        static_cast<uint8_t*>(host_buffers_[active_buffer_idx_]) + host_offset;

    CUDA_CHECK(
        cudaMemcpyAsync(target_host_buffer, gpu_ptr, bytes, cudaMemcpyDeviceToHost, copy_stream_));
}

void AsyncOffloader::write_host_to_nvme_async(const std::string& filepath, size_t bytes,
                                              size_t host_offset) {
    int fd = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Failed to open output file for writing: " + filepath);
    }

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        close(fd);
        throw std::runtime_error("io_uring submission queue is full.");
    }

    // Offset source memory address by host_offset
    uint8_t* source_host_buffer =
        static_cast<uint8_t*>(host_buffers_[active_buffer_idx_]) + host_offset;

    io_uring_prep_write(sqe, fd, source_host_buffer, bytes, 0);

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        close(fd);
        throw std::runtime_error("Failed to submit io_uring write request.");
    }

    active_fds_.push_back(fd);
    pending_writes_++;
}

void AsyncOffloader::read_nvme_to_host_async(const std::string& filepath, size_t bytes,
                                             size_t host_offset) {
    int fd = open(filepath.c_str(), O_RDONLY, 0644);
    if (fd < 0) {
        throw std::runtime_error("Failed to open input file for reading: " + filepath);
    }

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        close(fd);
        throw std::runtime_error("io_uring submission queue is full.");
    }

    uint8_t* target_host_buffer =
        static_cast<uint8_t*>(host_buffers_[active_buffer_idx_]) + host_offset;

    // Prep read request from NVMe to host buffer
    io_uring_prep_read(sqe, fd, target_host_buffer, bytes, 0);

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        close(fd);
        throw std::runtime_error("Failed to submit io_uring read request.");
    }

    active_fds_.push_back(fd);
    pending_reads_++;
}

void AsyncOffloader::load_host_to_gpu_async(uintptr_t gpu_ptr_address, size_t bytes,
                                            size_t host_offset) {
    if (host_offset + bytes > single_buffer_capacity_) {
        throw std::invalid_argument("Transfer size exceeds allocated single buffer capacity.");
    }

    void* gpu_ptr = reinterpret_cast<void*>(gpu_ptr_address);
    const uint8_t* source_host_buffer =
        static_cast<const uint8_t*>(host_buffers_[active_buffer_idx_]) + host_offset;

    // Copy from pinned host buffer back to GPU memory asynchronously
    CUDA_CHECK(
        cudaMemcpyAsync(gpu_ptr, source_host_buffer, bytes, cudaMemcpyHostToDevice, copy_stream_));
}

void AsyncOffloader::wait_gpu_to_host_completion() {
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
}

void AsyncOffloader::flush_nvme_writes() {
    while (pending_writes_ > 0) {
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret == 0) {
            if (cqe->res < 0) {
                std::cerr << "io_uring write failed with error code: " << cqe->res << std::endl;
            }
            io_uring_cqe_seen(&ring_, cqe);
            pending_writes_--;
        }
    }

    // Safely close file descriptors after writes complete
    for (int fd : active_fds_) {
        close(fd);
    }
    active_fds_.clear();
}
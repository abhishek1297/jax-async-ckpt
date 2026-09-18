#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "async_offloader.hpp"

namespace nb = nanobind;

NB_MODULE(_core_cpp, m) {
    nb::class_<AsyncOffloader>(m, "AsyncOffloader")
        .def(nb::init<size_t, unsigned>(), nb::arg("max_buffer_bytes"),
             nb::arg("io_ring_entries") = 64)
        .def("swap_buffers", &AsyncOffloader::swap_buffers)
        // Offload (Save) Methods
        .def("offload_gpu_to_host_async", &AsyncOffloader::offload_gpu_to_host_async,
             nb::arg("gpu_ptr_address"), nb::arg("bytes"), nb::arg("host_offset") = 0)
        .def("write_host_to_nvme_async", &AsyncOffloader::write_host_to_nvme_async,
             nb::arg("filepath"), nb::arg("bytes"), nb::arg("host_offset") = 0)
        // Restore Methods
        .def("read_nvme_to_host_async", &AsyncOffloader::read_nvme_to_host_async,
             nb::arg("filepath"), nb::arg("bytes"), nb::arg("host_offset") = 0)
        .def("load_host_to_gpu_async", &AsyncOffloader::load_host_to_gpu_async,
             nb::arg("gpu_ptr_address"), nb::arg("bytes"), nb::arg("host_offset") = 0)
        // Fences
        .def("wait_gpu_to_host_completion", &AsyncOffloader::wait_gpu_to_host_completion)
        .def("flush_nvme_writes", &AsyncOffloader::flush_nvme_writes);
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "replay_buffer.cpp"
#include <tuple>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(replay_buffer, m) {
    using BufferType = std::tuple<std::vector<float>, int, float, std::vector<float>, bool>;
    using ReplayBufferType = ReplayBuffer<BufferType>;

    py::class_<ReplayBufferType>(m, "ReplayBuffer")
        .def(py::init<size_t>())
        .def("add", &ReplayBufferType::add)
        .def("sample", &ReplayBufferType::sample)
        .def("random_batch", &ReplayBufferType::random_batch)
        .def("isEmpty", &ReplayBufferType::isEmpty)
        .def("size", &ReplayBufferType::size);
}


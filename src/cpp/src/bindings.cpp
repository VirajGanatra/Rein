#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "replay_buffer.cpp"
#include <tuple>
#include <vector>
#include "ppo_core.cpp"

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

PYBIND11_MODULE(ppo_core, m) {
    m.def("compute_advantages", &compute_advantages, "Compute advantages",
          py::arg("values"), py::arg("next_values"), py::arg("rewards"), py::arg("dones"), py::arg("gamma"), py::arg("gae_lambda"));
    m.def("compute_actor_loss", &compute_actor_loss, "Compute actor loss",
          py::arg("action_log_probs"), py::arg("old_action_log_probs"), py::arg("advantages"), py::arg("clip_param"));
    m.def("compute_critic_loss", &compute_critic_loss, "Compute critic loss",
          py::arg("values"), py::arg("returns"));
    m.def("compute_entropy_loss", &compute_entropy_loss, "Compute entropy loss",
          py::arg("action_probs"));
}
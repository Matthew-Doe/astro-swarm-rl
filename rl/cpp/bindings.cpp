#include "simulator.hpp"

#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using astroswarm::rl::BatchedSimulator;
using astroswarm::rl::Profile;
using astroswarm::rl::SimulatorConfig;

namespace {

SimulatorConfig config_from_dict(const py::dict& values) {
    SimulatorConfig config;
    auto get = [&values](const char* key, auto fallback) {
        return values.contains(key) ? values[key].cast<decltype(fallback)>() : fallback;
    };
    config.num_envs = get("num_envs", config.num_envs);
    config.threads = get("threads", config.threads);
    const auto profile = get("profile", std::string("balanced_self_play"));
    if (profile == "balanced_self_play") config.profile = Profile::Balanced;
    else if (profile == "farp_parity") config.profile = Profile::FarpParity;
    else throw std::invalid_argument("profile must be balanced_self_play or farp_parity");
    config.action_repeat = get("action_repeat", config.action_repeat);
    config.episode_seconds = get("episode_seconds", config.episode_seconds);
    config.min_defenders = get("min_defenders", config.min_defenders);
    config.max_defenders = get("max_defenders", config.max_defenders);
    config.primary_five_mix = get("primary_five_mix", config.primary_five_mix);
    config.collisions = get("collisions", config.collisions);
    config.defender_fov_min_degrees = get("defender_fov_min_degrees", config.defender_fov_min_degrees);
    config.defender_fov_max_degrees = get("defender_fov_max_degrees", config.defender_fov_max_degrees);
    config.defender_camp_radius = get("defender_camp_radius", config.defender_camp_radius);
    config.defender_camp_penalty = get("defender_camp_penalty", config.defender_camp_penalty);
    config.detection_reward = get("detection_reward", config.detection_reward);
    config.capture_reward = get("capture_reward", config.capture_reward);
    config.tracking_reward = get("tracking_reward", config.tracking_reward);
    config.closing_reward = get("closing_reward", config.closing_reward);
    config.seed = get("seed", config.seed);
    return config;
}

template <typename T>
py::array vector_view(std::vector<T>& values, std::vector<py::ssize_t> shape, py::object owner) {
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = static_cast<py::ssize_t>(sizeof(T));
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return py::array(py::buffer_info(values.data(), sizeof(T), py::format_descriptor<T>::format(),
                                     static_cast<py::ssize_t>(shape.size()), shape, strides), owner);
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "AstroSwarm high-throughput batched simulator";
    module.attr("MAX_DEFENDERS") = astroswarm::rl::kMaxDefenders;
    module.attr("INTERCEPTOR_OBS_DIM") = astroswarm::rl::kInterceptorObsDim;
    module.attr("ATTACKER_OBS_DIM") = astroswarm::rl::kAttackerObsDim;
    module.attr("SELF_DIM") = astroswarm::rl::kSelfDim;
    module.attr("INTERCEPTOR_ENTITY_DIM") = astroswarm::rl::kInterceptorEntityDim;
    module.attr("ATTACKER_ENTITY_DIM") = astroswarm::rl::kAttackerEntityDim;

    py::class_<BatchedSimulator>(module, "BatchedSimulator")
        .def(py::init([](const py::dict& config) { return BatchedSimulator(config_from_dict(config)); }))
        .def_property_readonly("num_envs", &BatchedSimulator::num_envs)
        .def_property_readonly("physics_ticks", &BatchedSimulator::physics_ticks)
        .def("reset", [](BatchedSimulator& self, std::uint64_t seed) {
            self.reset_all(seed);
        }, py::arg("seed"))
        .def("reset_done", &BatchedSimulator::reset_done)
        .def("reset_mask", [](BatchedSimulator& self,
                              py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask) {
            self.reset_mask(mask.data(), static_cast<std::size_t>(mask.size()));
        })
        .def("step", [](BatchedSimulator& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> interceptors,
                        py::array_t<float, py::array::c_style | py::array::forcecast> attacker) {
            py::gil_scoped_release release;
            self.step(interceptors.data(), static_cast<std::size_t>(interceptors.size()),
                      attacker.data(), static_cast<std::size_t>(attacker.size()));
        })
        .def("arrays", [](BatchedSimulator& self) {
            py::object owner = py::cast(&self, py::return_value_policy::reference);
            const auto n = static_cast<py::ssize_t>(self.num_envs());
            py::dict result;
            result["interceptor_obs"] = vector_view(const_cast<std::vector<float>&>(self.interceptor_obs()),
                {n, 6, static_cast<py::ssize_t>(astroswarm::rl::kInterceptorObsDim)}, owner);
            result["attacker_obs"] = vector_view(const_cast<std::vector<float>&>(self.attacker_obs()),
                {n, static_cast<py::ssize_t>(astroswarm::rl::kAttackerObsDim)}, owner);
            result["defender_mask"] = vector_view(const_cast<std::vector<std::uint8_t>&>(self.defender_mask()), {n, 6}, owner);
            result["interceptor_reward"] = vector_view(const_cast<std::vector<float>&>(self.interceptor_reward()), {n}, owner);
            result["attacker_reward"] = vector_view(const_cast<std::vector<float>&>(self.attacker_reward()), {n}, owner);
            result["shaping_reward"] = vector_view(const_cast<std::vector<float>&>(self.shaping_reward()), {n}, owner);
            result["detection_reward"] = vector_view(const_cast<std::vector<float>&>(self.detection_reward()), {n}, owner);
            result["terminal_reward"] = vector_view(const_cast<std::vector<float>&>(self.terminal_reward()), {n}, owner);
            result["camp_reward"] = vector_view(const_cast<std::vector<float>&>(self.camp_reward()), {n}, owner);
            result["tracking_reward"] = vector_view(const_cast<std::vector<float>&>(self.tracking_reward()), {n}, owner);
            result["closing_reward"] = vector_view(const_cast<std::vector<float>&>(self.closing_reward()), {n}, owner);
            result["terminated"] = vector_view(const_cast<std::vector<std::uint8_t>&>(self.terminated()), {n}, owner);
            result["truncated"] = vector_view(const_cast<std::vector<std::uint8_t>&>(self.truncated()), {n}, owner);
            result["outcome"] = vector_view(const_cast<std::vector<std::uint8_t>&>(self.outcome()), {n}, owner);
            result["detected"] = vector_view(const_cast<std::vector<std::uint8_t>&>(self.detected()), {n}, owner);
            result["elapsed_ticks"] = vector_view(const_cast<std::vector<std::uint32_t>&>(self.elapsed_ticks()), {n}, owner);
            return result;
        })
        .def("state", [](BatchedSimulator& self) {
            py::object owner = py::cast(&self, py::return_value_policy::reference);
            const auto n = static_cast<py::ssize_t>(self.num_envs());
            py::dict result;
            result["defender_x"] = vector_view(const_cast<std::vector<float>&>(self.defender_x()), {n, 6}, owner);
            result["defender_y"] = vector_view(const_cast<std::vector<float>&>(self.defender_y()), {n, 6}, owner);
            result["defender_heading"] = vector_view(const_cast<std::vector<float>&>(self.defender_heading()), {n, 6}, owner);
            result["attacker_x"] = vector_view(const_cast<std::vector<float>&>(self.attacker_x()), {n}, owner);
            result["attacker_y"] = vector_view(const_cast<std::vector<float>&>(self.attacker_y()), {n}, owner);
            result["attacker_heading"] = vector_view(const_cast<std::vector<float>&>(self.attacker_heading()), {n}, owner);
            return result;
        });
}

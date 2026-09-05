#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace astroswarm::rl {

inline constexpr std::size_t kMaxDefenders = 6;
inline constexpr std::size_t kActionDim = 2;
inline constexpr std::size_t kSelfDim = 12;
inline constexpr std::size_t kInterceptorEntityDim = 11;
inline constexpr std::size_t kInterceptorObsDim = kSelfDim + kMaxDefenders * kInterceptorEntityDim;
inline constexpr std::size_t kAttackerEntityDim = 13;
inline constexpr std::size_t kAttackerObsDim = kSelfDim + kMaxDefenders * kAttackerEntityDim;

enum class Profile : std::uint8_t { Balanced, FarpParity };
enum class Outcome : std::uint8_t { Active, InterceptorsWin, AttackerGoal, Timeout };

struct SimulatorConfig {
    std::size_t num_envs = 16384;
    std::size_t threads = 8;
    Profile profile = Profile::Balanced;
    std::uint32_t action_repeat = 4;
    float episode_seconds = 30.0F;
    std::uint32_t min_defenders = 1;
    std::uint32_t max_defenders = 6;
    bool primary_five_mix = true;
    bool collisions = false;
    float defender_fov_min_degrees = 35.0F;
    float defender_fov_max_degrees = 75.0F;
    float defender_camp_radius = 180.0F;
    float defender_camp_penalty = 0.001F;
    float detection_reward = 0.05F;
    float capture_reward = 1.0F;
    float tracking_reward = 0.0F;
    float closing_reward = 0.0F;
    std::uint64_t seed = 1;
};

class BatchedSimulator {
public:
    explicit BatchedSimulator(SimulatorConfig config);

    void reset_all(std::uint64_t seed);
    void reset_done();
    void reset_mask(const std::uint8_t* mask, std::size_t count);
    void step(const float* interceptor_actions, std::size_t interceptor_count,
              const float* attacker_actions, std::size_t attacker_count);

    [[nodiscard]] const SimulatorConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t num_envs() const noexcept { return config_.num_envs; }
    [[nodiscard]] std::size_t physics_ticks() const noexcept { return physics_ticks_; }

    [[nodiscard]] const std::vector<float>& interceptor_obs() const noexcept { return interceptor_obs_; }
    [[nodiscard]] const std::vector<float>& attacker_obs() const noexcept { return attacker_obs_; }
    [[nodiscard]] const std::vector<std::uint8_t>& defender_mask() const noexcept { return defender_mask_; }
    [[nodiscard]] const std::vector<float>& interceptor_reward() const noexcept { return interceptor_reward_; }
    [[nodiscard]] const std::vector<float>& attacker_reward() const noexcept { return attacker_reward_; }
    // Reward decomposition is attacker-centric; interceptor values are its negation.
    [[nodiscard]] const std::vector<float>& shaping_reward() const noexcept { return shaping_reward_; }
    [[nodiscard]] const std::vector<float>& detection_reward() const noexcept { return detection_reward_; }
    [[nodiscard]] const std::vector<float>& terminal_reward() const noexcept { return terminal_reward_; }
    [[nodiscard]] const std::vector<float>& camp_reward() const noexcept { return camp_reward_; }
    [[nodiscard]] const std::vector<float>& tracking_reward() const noexcept { return tracking_reward_; }
    [[nodiscard]] const std::vector<float>& closing_reward() const noexcept { return closing_reward_; }
    [[nodiscard]] const std::vector<std::uint8_t>& terminated() const noexcept { return terminated_; }
    [[nodiscard]] const std::vector<std::uint8_t>& truncated() const noexcept { return truncated_; }
    [[nodiscard]] const std::vector<std::uint8_t>& outcome() const noexcept { return outcome_; }
    [[nodiscard]] const std::vector<std::uint8_t>& detected() const noexcept { return detected_; }
    [[nodiscard]] const std::vector<std::uint32_t>& elapsed_ticks() const noexcept { return elapsed_ticks_; }

    [[nodiscard]] const std::vector<float>& defender_x() const noexcept { return defender_x_; }
    [[nodiscard]] const std::vector<float>& defender_y() const noexcept { return defender_y_; }
    [[nodiscard]] const std::vector<float>& defender_heading() const noexcept { return defender_heading_; }
    [[nodiscard]] const std::vector<float>& attacker_x() const noexcept { return attacker_x_; }
    [[nodiscard]] const std::vector<float>& attacker_y() const noexcept { return attacker_y_; }
    [[nodiscard]] const std::vector<float>& attacker_heading() const noexcept { return attacker_heading_; }

private:
    SimulatorConfig config_;
    std::size_t physics_ticks_ = 0;

    std::vector<float> defender_x_, defender_y_, defender_heading_, defender_cos_, defender_sin_;
    std::vector<float> defender_vx_, defender_vy_, defender_speed_, defender_turn_;
    std::vector<float> defender_view_, defender_fov_, defender_fov_cos_half_, defender_prev_throttle_, defender_prev_turn_;
    std::vector<float> attacker_x_, attacker_y_, attacker_heading_, attacker_cos_, attacker_sin_, attacker_vx_, attacker_vy_;
    std::vector<float> attacker_speed_, attacker_turn_, attacker_prev_throttle_, attacker_prev_turn_;
    std::vector<std::uint8_t> defender_mask_, defender_sees_attacker_;
    std::vector<std::uint8_t> detected_, terminated_, truncated_, outcome_;
    std::vector<std::uint32_t> elapsed_ticks_, episode_index_;
    std::vector<std::uint64_t> rng_state_, rng_stream_;
    std::vector<float> interceptor_reward_, attacker_reward_;
    std::vector<float> shaping_reward_, detection_reward_, terminal_reward_, camp_reward_, tracking_reward_, closing_reward_;
    std::vector<float> interceptor_obs_, attacker_obs_;

    [[nodiscard]] std::size_t di(std::size_t env, std::size_t defender) const noexcept;
    [[nodiscard]] std::uint32_t random_u32(std::size_t env) noexcept;
    [[nodiscard]] float random_unit(std::size_t env) noexcept;
    [[nodiscard]] float random_range(std::size_t env, float low, float high) noexcept;
    void seed_env(std::size_t env, std::uint64_t seed) noexcept;
    void reset_env(std::size_t env) noexcept;
    void step_env(std::size_t env, const float* interceptor_actions, const float* attacker_actions) noexcept;
    void build_observations() noexcept;
    void build_observation(std::size_t env) noexcept;
    [[nodiscard]] bool visible(std::size_t env, std::size_t observer, float target_x, float target_y) const noexcept;
    [[nodiscard]] float attacker_potential(std::size_t env) const noexcept;
};

}  // namespace astroswarm::rl

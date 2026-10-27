#include "simulator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace astroswarm::rl {
namespace {

constexpr float kArenaWidth = 3840.0F;
constexpr float kArenaHeight = 2160.0F;
constexpr float kPlanetX = 1920.0F;
constexpr float kPlanetY = 1080.0F;
constexpr float kPlanetRadius = 120.0F;
constexpr float kGoalRadius = 136.0F;
constexpr float kSpawnRadius = 1000.0F;
constexpr float kDefenderRadius = 9.0F;
constexpr float kAttackerRadius = 9.0F;
constexpr float kCaptureRadius = kDefenderRadius + kAttackerRadius;
constexpr float kPlacementMin = 170.0F;
constexpr float kPlacementMax = 250.0F;
constexpr float kPlacementSpacing = 110.0F;
constexpr float kDt = 1.0F / 60.0F;
constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kTau = 2.0F * kPi;

float clamp_action(float value) noexcept {
    return std::clamp(value, -1.0F, 1.0F);
}

float wrap_angle(float angle) noexcept {
    angle = std::fmod(angle + kPi, kTau);
    if (angle < 0.0F) angle += kTau;
    return angle - kPi;
}

float distance(float ax, float ay, float bx, float by) noexcept {
    return std::hypot(ax - bx, ay - by);
}

}  // namespace

BatchedSimulator::BatchedSimulator(SimulatorConfig config) : config_(std::move(config)) {
    if (config_.num_envs == 0) throw std::invalid_argument("num_envs must be positive");
    if (config_.threads == 0) config_.threads = 1;
    if (config_.action_repeat == 0) throw std::invalid_argument("action_repeat must be positive");
    if (config_.min_defenders < 1 || config_.max_defenders > kMaxDefenders ||
        config_.min_defenders > config_.max_defenders) {
        throw std::invalid_argument("defender bounds must satisfy 1 <= min <= max <= 6");
    }
    if (config_.defender_fov_min_degrees <= 0.0F ||
        config_.defender_fov_min_degrees > config_.defender_fov_max_degrees ||
        config_.defender_fov_max_degrees > 180.0F) {
        throw std::invalid_argument("defender FOV bounds must satisfy 0 < min <= max <= 180 degrees");
    }
    if (config_.defender_camp_radius < 0.0F || config_.defender_camp_penalty < 0.0F ||
        config_.detection_reward < 0.0F || config_.capture_reward < 0.0F ||
        config_.tracking_reward < 0.0F || config_.closing_reward < 0.0F ||
        config_.clean_goal_reward < 0.0F || config_.defender_camp_grace_seconds < 0.0F ||
        config_.tracking_reward_cap < 0.0F || config_.timeout_reward < 0.0F) {
        throw std::invalid_argument("reward magnitudes and camp radius must be non-negative");
    }

    if (!(config_.reward_shaping_gamma > 0.0F && config_.reward_shaping_gamma <= 1.0F)) {
        throw std::invalid_argument("reward_shaping_gamma must be in (0, 1]");
    }
    const auto envs = config_.num_envs;
    const auto defenders = envs * kMaxDefenders;
    defender_x_.resize(defenders); defender_y_.resize(defenders); defender_heading_.resize(defenders);
    defender_cos_.resize(defenders); defender_sin_.resize(defenders);
    defender_vx_.resize(defenders); defender_vy_.resize(defenders); defender_speed_.resize(defenders);
    defender_turn_.resize(defenders); defender_view_.resize(defenders); defender_fov_.resize(defenders);
    defender_fov_cos_half_.resize(defenders);
    defender_prev_throttle_.resize(defenders); defender_prev_turn_.resize(defenders);
    defender_idle_ticks_.resize(defenders);
    tracking_reward_paid_.resize(envs);
    defender_mask_.resize(defenders); defender_sees_attacker_.resize(defenders);
    attacker_x_.resize(envs); attacker_y_.resize(envs); attacker_heading_.resize(envs);
    attacker_cos_.resize(envs); attacker_sin_.resize(envs);
    attacker_vx_.resize(envs); attacker_vy_.resize(envs); attacker_speed_.resize(envs); attacker_turn_.resize(envs);
    attacker_prev_throttle_.resize(envs); attacker_prev_turn_.resize(envs);
    last_attacker_dx_.resize(defenders); last_attacker_dy_.resize(defenders);
    last_attacker_vx_.resize(defenders); last_attacker_vy_.resize(defenders);
    last_attacker_age_.resize(defenders);
    detected_.resize(envs); terminated_.resize(envs); truncated_.resize(envs); outcome_.resize(envs);
    elapsed_ticks_.resize(envs); episode_index_.resize(envs); rng_state_.resize(envs); rng_stream_.resize(envs);
    interceptor_reward_.resize(envs); attacker_reward_.resize(envs);
    shaping_reward_.resize(envs); detection_reward_.resize(envs); terminal_reward_.resize(envs); camp_reward_.resize(envs);
    tracking_reward_.resize(envs); closing_reward_.resize(envs);
    interceptor_obs_.resize(envs * kMaxDefenders * kInterceptorObsDim);
    attacker_obs_.resize(envs * kAttackerObsDim);
    reset_all(config_.seed);
}

std::size_t BatchedSimulator::di(std::size_t env, std::size_t defender) const noexcept {
    return env * kMaxDefenders + defender;
}

void BatchedSimulator::seed_env(std::size_t env, std::uint64_t seed) noexcept {
    rng_state_[env] = 0U;
    rng_stream_[env] = ((seed + env * 0x9E3779B97F4A7C15ULL) << 1U) | 1U;
    random_u32(env);
    rng_state_[env] += seed ^ (env * 0xD1B54A32D192ED03ULL);
    random_u32(env);
}

std::uint32_t BatchedSimulator::random_u32(std::size_t env) noexcept {
    const std::uint64_t old = rng_state_[env];
    rng_state_[env] = old * 6364136223846793005ULL + rng_stream_[env];
    const auto xorshifted = static_cast<std::uint32_t>(((old >> 18U) ^ old) >> 27U);
    const auto rotation = static_cast<std::uint32_t>(old >> 59U);
    return (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31U));
}

float BatchedSimulator::random_unit(std::size_t env) noexcept {
    return static_cast<float>(random_u32(env) >> 8U) * (1.0F / 16777216.0F);
}

float BatchedSimulator::random_range(std::size_t env, float low, float high) noexcept {
    return low + (high - low) * random_unit(env);
}

void BatchedSimulator::reset_all(std::uint64_t seed) {
    config_.seed = seed;
    physics_ticks_ = 0;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(config_.threads))
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(config_.num_envs); ++e) {
        episode_index_[e] = 0;
        seed_env(static_cast<std::size_t>(e), seed);
        reset_env(static_cast<std::size_t>(e));
    }
    build_observations();
}

void BatchedSimulator::reset_done() {
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(config_.threads))
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(config_.num_envs); ++e) {
        if (terminated_[e] || truncated_[e]) reset_env(static_cast<std::size_t>(e));
    }
    build_observations();
}

void BatchedSimulator::reset_mask(const std::uint8_t* mask, std::size_t count) {
    if (count != config_.num_envs) throw std::invalid_argument("reset mask has wrong size");
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(config_.threads))
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(config_.num_envs); ++e) {
        if (mask[e]) reset_env(static_cast<std::size_t>(e));
    }
    build_observations();
}

void BatchedSimulator::reset_env(std::size_t env) noexcept {
    if (episode_index_[env] > 0) {
        seed_env(env, config_.seed + static_cast<std::uint64_t>(episode_index_[env]) * 0xA24BAED4963EE407ULL);
    }
    ++episode_index_[env];
    tracking_reward_paid_[env] = 0.0F;
    elapsed_ticks_[env] = 0;
    detected_[env] = 0; terminated_[env] = 0; truncated_[env] = 0;
    outcome_[env] = static_cast<std::uint8_t>(Outcome::Active);
    interceptor_reward_[env] = 0.0F; attacker_reward_[env] = 0.0F;
    shaping_reward_[env] = 0.0F; detection_reward_[env] = 0.0F; terminal_reward_[env] = 0.0F; camp_reward_[env] = 0.0F;
    tracking_reward_[env] = 0.0F; closing_reward_[env] = 0.0F;
    attacker_prev_throttle_[env] = 0.0F; attacker_prev_turn_[env] = 0.0F;

    std::uint32_t count;
    if (config_.profile == Profile::FarpParity) {
        count = std::clamp<std::uint32_t>(5, config_.min_defenders, config_.max_defenders);
        attacker_speed_[env] = 150.0F;
        attacker_turn_[env] = 2.8F;
    } else {
        const bool use_five = config_.primary_five_mix && random_unit(env) < 0.70F &&
                              config_.min_defenders <= 5 && config_.max_defenders >= 5;
        count = use_five ? 5U : config_.min_defenders + random_u32(env) %
                                     (config_.max_defenders - config_.min_defenders + 1U);
        attacker_speed_[env] = random_range(env, 135.0F, 165.0F);
        attacker_turn_[env] = random_range(env, 2.5F, 3.1F);
    }

    for (std::size_t d = 0; d < kMaxDefenders; ++d) {
        const auto index = di(env, d);
        defender_idle_ticks_[index] = 0;
        defender_mask_[index] = d < count ? 1 : 0;
        defender_sees_attacker_[index] = 0;
        defender_prev_throttle_[index] = 0.0F;
        defender_prev_turn_[index] = 0.0F;
        last_attacker_dx_[index] = last_attacker_dy_[index] = 0.0F;
        last_attacker_vx_[index] = last_attacker_vy_[index] = 0.0F;
        last_attacker_age_[index] = 0;
        defender_vx_[index] = 0.0F; defender_vy_[index] = 0.0F;
        if (d >= count) {
            defender_x_[index] = defender_y_[index] = defender_heading_[index] = 0.0F;
            defender_cos_[index] = 1.0F; defender_sin_[index] = 0.0F;
            defender_speed_[index] = defender_turn_[index] = 0.0F;
            defender_view_[index] = defender_fov_[index] = 0.0F;
            continue;
        }

        float x = kPlanetX;
        float y = kPlanetY;
        for (int attempt = 0; attempt < 40; ++attempt) {
            const float angle = random_range(env, -kPi, kPi);
            const float radius = random_range(env, kPlacementMin, kPlacementMax);
            x = kPlanetX + std::cos(angle) * radius;
            y = kPlanetY + std::sin(angle) * radius;
            bool clear = true;
            for (std::size_t prior = 0; prior < d; ++prior) {
                const auto pi = di(env, prior);
                clear = clear && distance(x, y, defender_x_[pi], defender_y_[pi]) >= kPlacementSpacing;
            }
            if (clear) break;
        }
        defender_x_[index] = x; defender_y_[index] = y;
        defender_heading_[index] = random_range(env, -kPi, kPi);
        defender_cos_[index] = std::cos(defender_heading_[index]);
        defender_sin_[index] = std::sin(defender_heading_[index]);
        if (config_.profile == Profile::FarpParity) {
            defender_speed_[index] = 136.0F;
            defender_turn_[index] = 3.2F;
            defender_view_[index] = 180.0F;
            defender_fov_[index] = 50.0F * kPi / 180.0F;
        } else {
            defender_speed_[index] = random_range(env, 130.0F, 180.0F);
            defender_turn_[index] = random_range(env, 2.8F, 3.8F);
            const float sensor_budget = random_unit(env);
            defender_view_[index] = 180.0F + 120.0F * sensor_budget;
            const float fov_degrees = config_.defender_fov_max_degrees -
                (config_.defender_fov_max_degrees - config_.defender_fov_min_degrees) * sensor_budget;
            defender_fov_[index] = fov_degrees * kPi / 180.0F;
        }
        defender_fov_cos_half_[index] = std::cos(defender_fov_[index] * 0.5F);
    }

    const float attacker_angle = random_range(env, -kPi, kPi);
    attacker_x_[env] = kPlanetX + std::cos(attacker_angle) * kSpawnRadius;
    attacker_y_[env] = kPlanetY + std::sin(attacker_angle) * kSpawnRadius;
    attacker_heading_[env] = wrap_angle(attacker_angle + kPi);
    attacker_cos_[env] = std::cos(attacker_heading_[env]);
    attacker_sin_[env] = std::sin(attacker_heading_[env]);
    attacker_vx_[env] = attacker_vy_[env] = 0.0F;
}

float BatchedSimulator::attacker_potential(std::size_t env) const noexcept {
    const float remaining = distance(attacker_x_[env], attacker_y_[env], kPlanetX, kPlanetY) - kGoalRadius;
    const float start_distance = kSpawnRadius - kGoalRadius;
    return 0.25F * std::clamp(1.0F - remaining / start_distance, -1.0F, 1.0F);
}

bool BatchedSimulator::visible(std::size_t env, std::size_t observer, float target_x, float target_y) const noexcept {
    const auto index = di(env, observer);
    if (!defender_mask_[index]) return false;
    const float dx = target_x - defender_x_[index];
    const float dy = target_y - defender_y_[index];
    const float d2 = dx * dx + dy * dy;
    const float view = defender_view_[index];
    if (d2 > view * view) return false;
    const float length = std::sqrt(std::max(d2, 1.0e-12F));
    const float dot = (defender_cos_[index] * dx + defender_sin_[index] * dy) / length;
    return dot >= defender_fov_cos_half_[index];
}

void BatchedSimulator::step(const float* interceptor_actions, std::size_t interceptor_count,
                            const float* attacker_actions, std::size_t attacker_count) {
    if (interceptor_count != config_.num_envs * kMaxDefenders * kActionDim) {
        throw std::invalid_argument("interceptor actions must have shape [num_envs, 6, 2]");
    }
    if (attacker_count != config_.num_envs * kActionDim) {
        throw std::invalid_argument("attacker actions must have shape [num_envs, 2]");
    }
    physics_ticks_ += config_.num_envs * config_.action_repeat;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(config_.threads))
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(config_.num_envs); ++e) {
        step_env(static_cast<std::size_t>(e), interceptor_actions, attacker_actions);
    }
    build_observations();
}

void BatchedSimulator::step_env(std::size_t env, const float* interceptor_actions,
                                const float* attacker_actions) noexcept {
    interceptor_reward_[env] = 0.0F;
    attacker_reward_[env] = 0.0F;
    shaping_reward_[env] = 0.0F;
    detection_reward_[env] = 0.0F;
    terminal_reward_[env] = 0.0F;
    camp_reward_[env] = 0.0F;
    tracking_reward_[env] = 0.0F;
    closing_reward_[env] = 0.0F;
    if (terminated_[env] || truncated_[env]) return;
    const float old_potential = attacker_potential(env);
    bool newly_detected = false;
    bool defenders_camping = false;
    float closest_distance_before = std::numeric_limits<float>::max();
    std::size_t active_defenders = 0;
    for (std::size_t d = 0; d < kMaxDefenders; ++d) {
        const auto index = di(env, d);
        if (!defender_mask_[index]) continue;
        ++active_defenders;
        closest_distance_before = std::min(
            closest_distance_before, distance(defender_x_[index], defender_y_[index], attacker_x_[env], attacker_y_[env]));
    }
    float tracked_fraction_sum = 0.0F;
    float camping_fraction_sum = 0.0F;

    for (std::uint32_t repeat = 0; repeat < config_.action_repeat; ++repeat) {
        for (std::size_t d = 0; d < kMaxDefenders; ++d) {
            const auto index = di(env, d);
            if (!defender_mask_[index]) continue;
            const auto ai = index * kActionDim;
            const float throttle = clamp_action(interceptor_actions[ai]);
            const float turn = clamp_action(interceptor_actions[ai + 1]);
            defender_prev_throttle_[index] = throttle;
            defender_prev_turn_[index] = turn;
            defender_heading_[index] = wrap_angle(defender_heading_[index] + turn * defender_turn_[index] * kDt);
            defender_cos_[index] = std::cos(defender_heading_[index]);
            defender_sin_[index] = std::sin(defender_heading_[index]);
            defender_vx_[index] = defender_cos_[index] * defender_speed_[index] * throttle;
            defender_vy_[index] = defender_sin_[index] * defender_speed_[index] * throttle;
            defender_x_[index] = std::clamp(defender_x_[index] + defender_vx_[index] * kDt, 14.0F, kArenaWidth - 14.0F);
            defender_y_[index] = std::clamp(defender_y_[index] + defender_vy_[index] * kDt, 14.0F, kArenaHeight - 14.0F);

            const float planet_dx = defender_x_[index] - kPlanetX;
            const float planet_dy = defender_y_[index] - kPlanetY;
            const float planet_distance = std::hypot(planet_dx, planet_dy);
            const float planet_minimum = kPlanetRadius + kDefenderRadius;
            if (planet_distance < planet_minimum) {
                const float safe_distance = std::max(planet_distance, 1.0e-6F);
                defender_x_[index] = kPlanetX + planet_dx / safe_distance * planet_minimum;
                defender_y_[index] = kPlanetY + planet_dy / safe_distance * planet_minimum;
            }
            defenders_camping = defenders_camping ||
                (planet_distance < config_.defender_camp_radius && std::abs(throttle) < 0.2F);

            if (config_.collisions) {
                float px = defender_x_[index] - kPlanetX;
                float py = defender_y_[index] - kPlanetY;
                float pd = std::hypot(px, py);
                const float minimum = kPlanetRadius + kDefenderRadius;
                if (pd < minimum) {
                    if (pd < 1.0e-6F) { px = 1.0F; py = 0.0F; pd = 1.0F; }
                    defender_x_[index] = kPlanetX + px / pd * minimum;
                    defender_y_[index] = kPlanetY + py / pd * minimum;
                }
                for (std::size_t other = 0; other < kMaxDefenders; ++other) {
                    if (other == d || !defender_mask_[di(env, other)]) continue;
                    const auto oi = di(env, other);
                    float dx = defender_x_[index] - defender_x_[oi];
                    float dy = defender_y_[index] - defender_y_[oi];
                    float dd = std::hypot(dx, dy);
                    if (dd < 2.0F * kDefenderRadius && dd > 1.0e-6F) {
                        defender_x_[index] = defender_x_[oi] + dx / dd * 2.0F * kDefenderRadius;
                        defender_y_[index] = defender_y_[oi] + dy / dd * 2.0F * kDefenderRadius;
                    }
                }
            }
        }

        const auto aa = env * kActionDim;
        const float attacker_throttle = clamp_action(attacker_actions[aa]);
        const float attacker_turn = clamp_action(attacker_actions[aa + 1]);
        attacker_prev_throttle_[env] = attacker_throttle;
        attacker_prev_turn_[env] = attacker_turn;
        attacker_heading_[env] = wrap_angle(attacker_heading_[env] + attacker_turn * attacker_turn_[env] * kDt);
        attacker_cos_[env] = std::cos(attacker_heading_[env]);
        attacker_sin_[env] = std::sin(attacker_heading_[env]);
        attacker_vx_[env] = attacker_cos_[env] * attacker_speed_[env] * attacker_throttle;
        attacker_vy_[env] = attacker_sin_[env] * attacker_speed_[env] * attacker_throttle;
        attacker_x_[env] = std::clamp(attacker_x_[env] + attacker_vx_[env] * kDt, 20.0F, kArenaWidth - 20.0F);
        attacker_y_[env] = std::clamp(attacker_y_[env] + attacker_vy_[env] * kDt, 20.0F, kArenaHeight - 20.0F);
        ++elapsed_ticks_[env];

        bool captured = false;
        std::size_t tracked_defenders = 0;
        std::size_t camping_defenders = 0;
        for (std::size_t d = 0; d < kMaxDefenders; ++d) {
            const auto index = di(env, d);
            if (!defender_mask_[index]) continue;
            const bool sees = visible(env, d, attacker_x_[env], attacker_y_[env]);
            defender_sees_attacker_[index] = sees ? 1 : 0;
            if (sees) {
                last_attacker_dx_[index] = (attacker_x_[env] - defender_x_[index]) / kSpawnRadius;
                last_attacker_dy_[index] = (attacker_y_[env] - defender_y_[index]) / kSpawnRadius;
                last_attacker_vx_[index] = (attacker_vx_[env] - defender_vx_[index]) / 400.0F;
                last_attacker_vy_[index] = (attacker_vy_[env] - defender_vy_[index]) / 400.0F;
                last_attacker_age_[index] = 0;
            } else if (last_attacker_age_[index] < 255U) {
                ++last_attacker_age_[index];
            }
            tracked_defenders += sees ? 1U : 0U;
            if (config_.defender_camp_conditional) {
                const bool idle = !sees && std::abs(defender_prev_throttle_[index]) < 0.2F &&
                    distance(defender_x_[index], defender_y_[index], kPlanetX, kPlanetY) < config_.defender_camp_radius;
                defender_idle_ticks_[index] = idle ? defender_idle_ticks_[index] + 1U : 0U;
                camping_defenders += static_cast<float>(defender_idle_ticks_[index]) >
                    config_.defender_camp_grace_seconds * 60.0F ? 1U : 0U;
            }
            if (sees && !detected_[env]) newly_detected = true;
            const float capture_dx = defender_x_[index] - attacker_x_[env];
            const float capture_dy = defender_y_[index] - attacker_y_[env];
            captured = captured || capture_dx * capture_dx + capture_dy * capture_dy <= kCaptureRadius * kCaptureRadius;
        }
        tracked_fraction_sum += config_.tracking_team_contact ? (tracked_defenders > 0 ? 1.0F : 0.0F) :
            static_cast<float>(tracked_defenders) / static_cast<float>(active_defenders);
        camping_fraction_sum += static_cast<float>(camping_defenders) / static_cast<float>(active_defenders);
        if (newly_detected) detected_[env] = 1;

        if (captured) {
            terminated_[env] = 1;
            outcome_[env] = static_cast<std::uint8_t>(Outcome::InterceptorsWin);
            interceptor_reward_[env] += config_.capture_reward;
            attacker_reward_[env] -= config_.capture_reward;
            terminal_reward_[env] -= config_.capture_reward;
            break;
        }
        const float goal_dx = attacker_x_[env] - kPlanetX;
        const float goal_dy = attacker_y_[env] - kPlanetY;
        if (goal_dx * goal_dx + goal_dy * goal_dy <= kGoalRadius * kGoalRadius) {
            terminated_[env] = 1;
            outcome_[env] = static_cast<std::uint8_t>(Outcome::AttackerGoal);
            // Detection is sticky for the episode, including the goal tick.
            const float goal_reward = detected_[env] ? 1.0F : config_.clean_goal_reward;
            attacker_reward_[env] += goal_reward;
            interceptor_reward_[env] -= goal_reward;
            terminal_reward_[env] += goal_reward;
            break;
        }
        if (elapsed_ticks_[env] >= static_cast<std::uint32_t>(std::ceil(config_.episode_seconds * 60.0F))) {
            truncated_[env] = 1;
            outcome_[env] = static_cast<std::uint8_t>(Outcome::Timeout);
            interceptor_reward_[env] += config_.timeout_reward;
            attacker_reward_[env] -= config_.timeout_reward;
            terminal_reward_[env] -= config_.timeout_reward;
            break;
        }
    }

    const float shaping = attacker_potential(env) - old_potential;
    shaping_reward_[env] = shaping;
    attacker_reward_[env] += shaping;
    interceptor_reward_[env] -= shaping;
    if (newly_detected) {
        detection_reward_[env] = -config_.detection_reward;
        attacker_reward_[env] -= config_.detection_reward;
        interceptor_reward_[env] += config_.detection_reward;
    }
    const float repeats = static_cast<float>(config_.action_repeat);
    camp_reward_[env] = config_.defender_camp_penalty * (config_.defender_camp_conditional ?
        camping_fraction_sum / repeats : (defenders_camping ? 1.0F : 0.0F));
    attacker_reward_[env] += camp_reward_[env];
    interceptor_reward_[env] -= camp_reward_[env];
    float tracking_payment = config_.tracking_reward * tracked_fraction_sum / repeats;
    if (config_.tracking_reward_cap > 0.0F) {
        tracking_payment = std::min(tracking_payment,
            std::max(0.0F, config_.tracking_reward_cap - tracking_reward_paid_[env]));
    }
    tracking_reward_paid_[env] += tracking_payment;
    tracking_reward_[env] = -tracking_payment;
    attacker_reward_[env] += tracking_reward_[env];
    interceptor_reward_[env] -= tracking_reward_[env];
    float closest_distance_after = std::numeric_limits<float>::max();
    for (std::size_t d = 0; d < kMaxDefenders; ++d) {
        const auto index = di(env, d);
        if (!defender_mask_[index]) continue;
        closest_distance_after = std::min(
            closest_distance_after, distance(defender_x_[index], defender_y_[index], attacker_x_[env], attacker_y_[env]));
    }
    const float closing_delta = (closest_distance_before - closest_distance_after) / kSpawnRadius;
    if (config_.closing_potential_shaping) {
        // Bounded team pursuit potential. Use PPO's decision-step discount and
        // zero potential for all terminal states (timeouts are terminal in GAE).
        const float before = 1.0F - std::clamp(closest_distance_before / kSpawnRadius, 0.0F, 1.0F);
        const float after = (terminated_[env] || truncated_[env]) ? 0.0F :
            1.0F - std::clamp(closest_distance_after / kSpawnRadius, 0.0F, 1.0F);
        closing_reward_[env] = -config_.closing_reward * (config_.reward_shaping_gamma * after - before);
    } else {
        closing_reward_[env] = -config_.closing_reward * std::clamp(closing_delta, -1.0F, 1.0F);
    }
    attacker_reward_[env] += closing_reward_[env];
    interceptor_reward_[env] -= closing_reward_[env];
}

void BatchedSimulator::build_observations() noexcept {
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(config_.threads))
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(config_.num_envs); ++e) {
        build_observation(static_cast<std::size_t>(e));
    }
}

void BatchedSimulator::build_observation(std::size_t env) noexcept {
    const float remaining = 1.0F - static_cast<float>(elapsed_ticks_[env]) /
        std::max(1.0F, config_.episode_seconds * 60.0F);
    std::uint32_t active_count = 0;
    for (std::size_t d = 0; d < kMaxDefenders; ++d) active_count += defender_mask_[di(env, d)];

    for (std::size_t d = 0; d < kMaxDefenders; ++d) {
        const auto index = di(env, d);
        const auto base = index * kInterceptorObsDim;
        auto* out = interceptor_obs_.data() + base;
        std::fill(out, out + kInterceptorObsDim, 0.0F);
        if (!defender_mask_[index]) continue;
        out[0] = (defender_x_[index] - kPlanetX) / kSpawnRadius;
        out[1] = (defender_y_[index] - kPlanetY) / kSpawnRadius;
        out[2] = defender_sin_[index]; out[3] = defender_cos_[index];
        out[4] = defender_x_[index] / kArenaWidth; out[5] = (kArenaWidth - defender_x_[index]) / kArenaWidth;
        out[6] = defender_y_[index] / kArenaHeight; out[7] = (kArenaHeight - defender_y_[index]) / kArenaHeight;
        out[8] = remaining; out[9] = static_cast<float>(active_count) / 6.0F;
        out[10] = defender_prev_throttle_[index]; out[11] = defender_prev_turn_[index];
        out[12] = last_attacker_dx_[index]; out[13] = last_attacker_dy_[index];
        out[14] = last_attacker_vx_[index]; out[15] = last_attacker_vy_[index];
        out[16] = std::min(1.0F, static_cast<float>(last_attacker_age_[index]) / 120.0F);

        for (std::size_t slot = 0; slot < kMaxDefenders; ++slot) {
            bool is_enemy = slot == 0;
            std::size_t other = 0;
            if (!is_enemy) {
                std::size_t seen = 0;
                for (std::size_t candidate = 0; candidate < kMaxDefenders; ++candidate) {
                    if (candidate == d) continue;
                    if (++seen == slot) { other = candidate; break; }
                }
            }
            float tx, ty, tvx, tvy;
            bool present;
            if (is_enemy) {
                tx = attacker_x_[env]; ty = attacker_y_[env]; tvx = attacker_vx_[env]; tvy = attacker_vy_[env];
                present = true;
            } else {
                const auto oi = di(env, other);
                tx = defender_x_[oi]; ty = defender_y_[oi]; tvx = defender_vx_[oi]; tvy = defender_vy_[oi];
                present = defender_mask_[oi] != 0;
            }
            const bool can_see = present && visible(env, d, tx, ty);
            if (!can_see) continue;
            const float dx = tx - defender_x_[index];
            const float dy = ty - defender_y_[index];
            const float dist = std::hypot(dx, dy);
            const float inverse_distance = 1.0F / std::max(dist, 1.0e-6F);
            const float bearing_sin = (defender_cos_[index] * dy - defender_sin_[index] * dx) * inverse_distance;
            const float bearing_cos = (defender_cos_[index] * dx + defender_sin_[index] * dy) * inverse_distance;
            const float target_sin = is_enemy ? attacker_sin_[env] : defender_sin_[di(env, other)];
            const float target_cos = is_enemy ? attacker_cos_[env] : defender_cos_[di(env, other)];
            const auto offset = kInterceptorSelfDim + slot * kInterceptorEntityDim;
            out[offset + 0] = dx / kSpawnRadius; out[offset + 1] = dy / kSpawnRadius;
            out[offset + 2] = (tvx - defender_vx_[index]) / 400.0F;
            out[offset + 3] = (tvy - defender_vy_[index]) / 400.0F;
            out[offset + 4] = target_sin * defender_cos_[index] - target_cos * defender_sin_[index];
            out[offset + 5] = target_cos * defender_cos_[index] + target_sin * defender_sin_[index];
            out[offset + 6] = dist / kSpawnRadius;
            out[offset + 7] = bearing_sin; out[offset + 8] = bearing_cos;
            out[offset + 9] = is_enemy ? 1.0F : -1.0F;
            out[offset + 10] = 1.0F;
        }
    }

    auto* out = attacker_obs_.data() + env * kAttackerObsDim;
    std::fill(out, out + kAttackerObsDim, 0.0F);
    out[0] = (attacker_x_[env] - kPlanetX) / kSpawnRadius;
    out[1] = (attacker_y_[env] - kPlanetY) / kSpawnRadius;
    out[2] = attacker_sin_[env]; out[3] = attacker_cos_[env];
    out[4] = attacker_x_[env] / kArenaWidth; out[5] = (kArenaWidth - attacker_x_[env]) / kArenaWidth;
    out[6] = attacker_y_[env] / kArenaHeight; out[7] = (kArenaHeight - attacker_y_[env]) / kArenaHeight;
    out[8] = remaining; out[9] = static_cast<float>(active_count) / 6.0F;
    out[10] = attacker_prev_throttle_[env]; out[11] = attacker_prev_turn_[env];
    for (std::size_t d = 0; d < kMaxDefenders; ++d) {
        const auto index = di(env, d);
        const auto offset = kSelfDim + d * kAttackerEntityDim;
        if (!defender_mask_[index]) continue;
        const float dx = defender_x_[index] - attacker_x_[env];
        const float dy = defender_y_[index] - attacker_y_[env];
        const float dist = std::hypot(dx, dy);
        const float inverse_distance = 1.0F / std::max(dist, 1.0e-6F);
        const float bearing_sin = (attacker_cos_[env] * dy - attacker_sin_[env] * dx) * inverse_distance;
        const float bearing_cos = (attacker_cos_[env] * dx + attacker_sin_[env] * dy) * inverse_distance;
        out[offset + 0] = dx / kSpawnRadius; out[offset + 1] = dy / kSpawnRadius;
        out[offset + 2] = (defender_vx_[index] - attacker_vx_[env]) / 400.0F;
        out[offset + 3] = (defender_vy_[index] - attacker_vy_[env]) / 400.0F;
        out[offset + 4] = defender_sin_[index] * attacker_cos_[env] - defender_cos_[index] * attacker_sin_[env];
        out[offset + 5] = defender_cos_[index] * attacker_cos_[env] + defender_sin_[index] * attacker_sin_[env];
        out[offset + 6] = dist / kSpawnRadius;
        out[offset + 7] = bearing_sin; out[offset + 8] = bearing_cos;
        out[offset + 9] = defender_view_[index] / 500.0F;
        out[offset + 10] = defender_fov_[index] / kPi;
        out[offset + 11] = defender_sees_attacker_[index] ? 1.0F : 0.0F;
        out[offset + 12] = 1.0F;
    }
}

}  // namespace astroswarm::rl

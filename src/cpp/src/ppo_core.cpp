#include <torch/torch.h>
#include <vector>

torch::Tensor compute_advantages(
    const torch::Tensor& values,
    const torch::Tensor& next_values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    float gamma,
    float gae_lambda) {

    torch::Tensor advantages = torch::zeros_like(values);
    torch::Tensor advantage = torch::zeros({1});

    for (int64_t t = values.size(0) - 1; t >= 0; --t) {
        auto delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t];
        advantage = delta + gamma * gae_lambda * (1 - dones[t]) * advantage;
        advantages[t] = advantage;
    }

    return advantages;
}

float compute_actor_loss(
    const torch::Tensor& action_log_probs,
    const torch::Tensor& old_action_log_probs,
    const torch::Tensor& advantages,
    float clip_param) {

    auto ratios = torch::exp(action_log_probs - old_action_log_probs);
    auto surr1 = ratios * advantages;
    auto surr2 = torch::clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages;
    auto actor_loss = -torch::min(surr1, surr2).mean();

    return actor_loss.item<float>();
}

float compute_critic_loss(
    const torch::Tensor& values,
    const torch::Tensor& returns) {

    auto value_loss = torch::mse_loss(values, returns);
    return value_loss.item<float>();
}

float compute_entropy_loss(
    const torch::Tensor& action_probs) {

    auto entropy_loss = -torch::sum(action_probs * torch::log(action_probs + 1e-8), -1).mean();
    return entropy_loss.item<float>();
}
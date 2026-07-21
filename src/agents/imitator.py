from transformer_mcts import transformer
from cg.api import to_observation_class


def imitate(
    obs_dict: dict,
    your_deck: list[int],
    chosen_action: list[int],
    expert_value: float,
    epsilon: float = 0.1,
) -> transformer.LearnSample:

    obs = to_observation_class(obs_dict)

    # 1. Generate all legal actions (extracted from mcts.create_node)
    actions = []
    indices = list(range(obs.select.maxCount))
    for _ in range(64):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < len(obs.select.option) - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break

    # 2. Map the expert action to the policy array with label smoothing
    num_actions = len(actions)

    # Ensure indices are sorted to match the order format of the combinatorial generator
    expert_action = sorted(chosen_action)

    if num_actions <= 1:
        policy = [1.0] * num_actions
    else:
        # Uniform distribution baseline for label smoothing
        policy = [epsilon / num_actions] * num_actions
        expert_idx = actions.index(expert_action)
        policy[expert_idx] += 1.0 - epsilon

        # Normalize to account for float precision
        total_p = sum(policy)
        policy = [p / total_p for p in policy]

    # 3. Create the SparseVectors for the network
    sv_enc = transformer.get_encoder_input(obs, your_deck)
    sv_dec = transformer.get_decoder_input(obs, actions)

    return transformer.LearnSample(
        value=expert_value, policy=policy, sv_enc=sv_enc, sv_dec=sv_dec
    )


class ImitationModel:
    """Wraps a rule-based expert bot to collect trajectories for imitation learning."""

    def __init__(self, expert_fn, deck, gamma=0.95, epsilon=0.1):
        self.expert_fn = expert_fn
        self.deck = deck
        self.gamma = gamma
        self.epsilon = epsilon
        self.trajectory = []

    def reset(self):
        self.trajectory = []

    def __call__(self, obs):
        # Call the rule-based agent function to fetch active action indices
        selected = self.expert_fn(obs)
        # Record structural state snapshot for backward processing pass
        self.trajectory.append((to_observation_class(obs), selected))
        return selected, None

    def post_process_samples(self, final_result):
        if not self.trajectory:
            return []

        samples = []
        total_steps = len(self.trajectory)
        player_index = self.trajectory[0][0].current.yourIndex

        for t, (obs, selected) in enumerate(self.trajectory):
            if final_result == 2:  # Draw
                base_reward = 0.0
            elif final_result == player_index:
                base_reward = 1.0
            else:
                base_reward = -1.0

            steps_to_end = total_steps - 1 - t
            discounted_value = base_reward * (self.gamma**steps_to_end)

            sample = build_imitation_sample(
                obs=obs,
                your_deck=self.deck,
                chosen_action=selected,
                expert_value=discounted_value,
                epsilon=self.epsilon,
            )
            samples.append(sample)
        return samples


def build_imitation_sample(obs, your_deck, chosen_action, expert_value, epsilon=0.1):
    """Generates a target training sample from expert action inputs."""
    actions = []
    indices = list(range(obs.select.maxCount))
    for _ in range(64):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < len(obs.select.option) - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break

    num_actions = len(actions)
    expert_action = sorted(chosen_action)

    if num_actions <= 1:
        policy = [1.0] * num_actions
    else:
        policy = [epsilon / num_actions] * num_actions
        try:
            expert_idx = actions.index(expert_action)
            policy[expert_idx] += 1.0 - epsilon
        except ValueError:
            policy[0] += 1.0 - epsilon

        total_p = sum(policy)
        policy = [p / total_p for p in policy]

    sv_enc = transformer.get_encoder_input(obs, your_deck)
    sv_dec = transformer.get_decoder_input(obs, actions)

    return transformer.LearnSample(
        value=expert_value, policy=policy, sv_enc=sv_enc, sv_dec=sv_dec
    )

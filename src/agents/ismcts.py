import math
import random
import typing
from collections import Counter

import torch

from cg.api import (
    SearchState,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from player import Player
from transformer_mcts import transformer
from utils.utils import get_public_cards

SEARCH_COUNT = 70


# MCTS Node Child
class Child:
    node: "Node | None"
    select: list[int]  # Selected option indices
    prob: float  # Probability

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob


# MCTS Node
class Node:
    value: float  # Self value
    total: float  # Total value
    visit: int  # Visit count
    parent: "Node | None"  # Parent node
    children: list[Child]
    state: SearchState  # Search State of this node

    def __init__(self, parent: "Node | None", state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    # Backpropagation value
    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def create_node(
    parent: Node | None,
    search_state: SearchState,
    your_index: int,
    your_deck: list[int],
    model: transformer.MyModel,
) -> tuple[Node, transformer.LearnSample | None]:
    node = Node(parent, search_state)

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        # Battle finished
        if state.result == 2:
            node.value = 0
        elif state.result == your_index:
            node.value = 1
        else:
            node.value = -1
        node.backprop(node.value)
        sample = None
    else:
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

        # Determine which deck to pass based on who is active in this simulation step
        if obs.current.yourIndex == your_index:
            current_deck = your_deck
        else:
            # Use the simulated Snorlax deck we initialized in search_begin.
            opponent_active_index = obs.current.yourIndex
            current_deck = [transformer.SENTINEL_UNK] * obs.current.players[
                opponent_active_index
            ].deckCount

        sv_enc = transformer.get_encoder_input(obs, current_deck)
        sv_dec = transformer.get_decoder_input(obs, actions)
        value, policy = transformer.eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)

        sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            sum += p
        for c in node.children:
            c.prob /= sum
        sample = transformer.LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)


class ISMCTSPlayer(Player):
    def __init__(
        self,
        model: typing.Callable,
        name: str,
        deck: list[int],
        sampler: typing.Any,
        sample_count: int = 10,
        search_count_per_sample: int = 10,
        c_puct: float = 0.4,
        is_trainable: bool = True,
    ):
        super().__init__(model=model, name=name, deck=deck, is_trainable=is_trainable)
        self.sampler = sampler
        self.sample_count = sample_count
        self.search_count_per_sample = search_count_per_sample
        self.c_puct = c_puct
        self.is_eval = False

    def reset(self):
        if hasattr(self.sampler, "reset"):
            self.sampler.reset()

    def process_obs(self, obs):
        if hasattr(self.sampler, "update"):
            self.sampler.update(obs)

    # We will perform exploration using MCTS and select actions. At the same time, we will also generate training data.
    def mcts_agent(
        self,
        obs_dict: dict,
        your_deck: list[int],
        model: transformer.MyModel,
        is_eval: bool = False,
    ) -> tuple[list[int], transformer.LearnSample]:
        obs = to_observation_class(obs_dict)
        your_index = obs.current.yourIndex
        state = obs.current

        aggregated_visits = {}
        aggregated_value = 0.0
        final_sample = None
        root_actions = []

        for sample_idx in range(self.sample_count):
            # 1. Sample decks/prizes
            your_public_cards = get_public_cards(player_state=state.players[your_index])
            stadium_card = state.stadium
            if stadium_card and stadium_card[0].playerIndex == your_index:
                your_public_cards.append(stadium_card.id)

            remaining = Counter(your_deck) - Counter(your_public_cards)
            your_sampled_deck = random.sample(
                remaining.elements(), state.players[your_index].deckCount
            )
            remaining -= Counter(your_sampled_deck)
            your_sampled_prize = random.sample(
                remaining.elements(), len(state.players[your_index].prize)
            )  # Fixed trailing comma

            opp_cards = self.sampler.sample_opponent(obs)

            search_state = search_begin(
                obs,
                your_deck=your_sampled_deck,
                your_prize=your_sampled_prize,
                opponent_deck=opp_cards["deck"],
                opponent_prize=opp_cards["prize"],
                opponent_hand=opp_cards["hand"],
                opponent_active=opp_cards["hidden_active"],
            )

            root, sample = create_node(None, search_state, your_index, your_deck, model)

            # Track actions on first pass
            if sample_idx == 0:
                final_sample = sample
                root_actions = [tuple(c.select) for c in root.children]
                for action in root_actions:
                    aggregated_visits[action] = 0

            # Apply Dirichlet noise to root children during self-play
            if not is_eval and len(root.children) > 1:
                DIRICHLET_ALPHA = 0.3
                EXPLORATION_FRACTION = 0.25
                noise = (
                    torch.distributions.dirichlet.Dirichlet(
                        torch.full((len(root.children),), DIRICHLET_ALPHA)
                    )
                    .sample()
                    .tolist()
                )
                for i, child in enumerate(root.children):
                    child.prob = (
                        child.prob * (1 - EXPLORATION_FRACTION)
                        + noise[i] * EXPLORATION_FRACTION
                    )

            # 2. Run MCTS iterations for this sample
            for _ in range(self.search_count_per_sample):
                current = root
                while True:
                    value = -1e9
                    c = self.c_puct * math.sqrt(current.visit)
                    for child in current.children:
                        visit = 0
                        if child.node is None:
                            v = current.total / current.visit
                        else:
                            v = child.node.total / child.node.visit
                            visit = child.node.visit
                        if current.state.observation.current.yourIndex != your_index:
                            v = -v
                        v += c * child.prob / (1 + visit)
                        if value < v:
                            value = v
                            next_child = child

                    if next_child.node is None:
                        step_state = search_step(
                            current.state.searchId, next_child.select
                        )
                        next_child.node, _ = create_node(
                            current, step_state, your_index, your_deck, model
                        )
                        break
                    else:
                        current = next_child.node
                        if current.state.observation.current.result >= 0:
                            current.backprop(current.value)
                            break

            # 3. Accumulate visits and root value across determinizations
            for child in root.children:
                if child.node is not None:
                    action_tuple = tuple(child.select)
                    if action_tuple in aggregated_visits:
                        aggregated_visits[action_tuple] += child.node.visit

            if root.visit > 0:
                aggregated_value += root.total / root.visit

            search_end()

        # 4. Final choice & policy target generation
        best_action = list(max(aggregated_visits, key=aggregated_visits.get))
        total_visits = sum(aggregated_visits.values())

        final_sample.value = aggregated_value / self.sample_count

        if total_visits > 0:
            for i, action in enumerate(root_actions):
                final_sample.policy[i] = aggregated_visits[action] / total_visits
        else:
            for i in range(len(root_actions)):
                final_sample.policy[i] = 1.0 / len(root_actions)

        return (best_action, final_sample)

    def __call__(self, obs):
        return self.mcts_agent(
            obs_dict=obs, your_deck=self.deck, model=self.model, is_eval=self.is_eval
        )

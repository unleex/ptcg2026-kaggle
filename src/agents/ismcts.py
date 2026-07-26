import math
import random
import typing

import torch

# Assuming your api wrappers and dataclasses are imported
from cg.api import (
    Observation,
    search_begin,
    search_end,
    search_step,
)
from player import Player
from transformer_mcts.mcts import create_node


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

    def __call__(self, obs: Observation) -> tuple[list[int], typing.Any]:
        self.process_obs(obs)

        your_index = obs.current.yourIndex
        state = obs.current
        your_state = state.players[your_index]

        aggregated_visits = {}
        aggregated_value = 0.0
        final_sample = None
        first_children_selects = []

        for _ in range(self.sample_count):
            # 1. Sample hidden cards
            your_deck_sampled = random.sample(self.deck, your_state.deckCount)
            your_prize_sampled = random.sample(self.deck, len(your_state.prize))

            opp_cards = self.sampler.sample_opponent(obs)
            # 2. Initialize simulator with current sample
            search_state = search_begin(
                obs,
                your_deck=your_deck_sampled,
                your_prize=your_prize_sampled,
                opponent_deck=opp_cards["deck"],
                opponent_prize=opp_cards["prize"],
                opponent_hand=opp_cards["hand"],
                opponent_active=opp_cards["hidden_active"],
            )

            root, sample = create_node(
                None, search_state, your_index, self.deck, self.model
            )

            # Setup tracking variables on the first sample pass
            if final_sample is None:
                final_sample = sample
                first_children_selects = [tuple(c.select) for c in root.children]
                for sel in first_children_selects:
                    aggregated_visits[sel] = 0

            # 3. Apply Dirichlet exploration noise (Train only)
            if not self.is_eval and len(root.children) > 1:
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

            # 4. Single-sample MCTS Loop
            for _ in range(self.search_count_per_sample):
                current = root
                while True:
                    value = -1e9
                    c = self.c_puct * math.sqrt(current.visit)
                    next_child = None
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
                            current, step_state, your_index, self.deck, self.model
                        )
                        break
                    else:
                        current = next_child.node
                        if current.state.observation.current.result >= 0:
                            current.backprop(current.value)
                            break

            # 5. Accumulate root visits and values for this determinization
            for child in root.children:
                if child.node is not None:
                    sel = tuple(child.select)
                    if sel in aggregated_visits:
                        aggregated_visits[sel] += child.node.visit

            if root.visit > 0:
                aggregated_value += root.total / root.visit

            # Free memory for this determinization session
            search_end()

        # 6. Final target generation and action selection
        best_select_tuple = max(aggregated_visits, key=aggregated_visits.get)
        best_select = list(best_select_tuple)

        # Average the value across all samples
        final_sample.value = aggregated_value / self.sample_count

        # Build the final policy target distribution
        sum_visits = sum(aggregated_visits.values())
        if sum_visits > 0:
            for i, sel in enumerate(first_children_selects):
                final_sample.policy[i] = aggregated_visits[sel] / sum_visits
        else:
            for i in range(len(first_children_selects)):
                final_sample.policy[i] = 1.0 / len(first_children_selects)

        return best_select, final_sample

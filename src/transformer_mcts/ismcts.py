import math
import random
import typing
from collections import Counter

import ray
import torch
from ray import serve

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
    opponents_deck: list[int],
    model: transformer.MyModel,
    batched_inference: bool = False,
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
        # FIXME: opponents_deck's size changes from the root node!
        # XXX: current fix is a crutch
        if obs.current.yourIndex == your_index:
            current_deck = your_deck
        else:
            current_deck = opponents_deck[
                : obs.current.players[obs.current.yourIndex].deckCount
            ]
            random.shuffle(current_deck)

        sv_enc = transformer.get_encoder_input(obs, current_deck)
        sv_dec = transformer.get_decoder_input(obs, actions)
        if batched_inference:
            value, policy = (
                serve.get_app_handle("model").remote((sv_enc, sv_dec)).result()
            )
        else:
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
        batched_inference: bool = False,
    ):
        super().__init__(model=model, name=name, deck=deck, is_trainable=is_trainable)
        self.sampler = sampler
        self.sample_count = sample_count
        self.search_count_per_sample = search_count_per_sample
        self.c_puct = c_puct
        self.is_eval = False
        self.batched_inference = batched_inference

    def reset(self):
        self.sampler.reset()

    def process_obs(self, obs):
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
        aggregated_value = 0
        aggregated_visits = {}
        for sample_idx in range(self.sample_count):
            your_public_cards = get_public_cards(player_state=state.players[your_index])
            stadium_card = state.stadium
            if stadium_card and stadium_card[0].playerIndex == your_index:
                your_public_cards.append(stadium_card[0].id)
            remaining = Counter(your_deck) - Counter(your_public_cards)
            your_sampled_deck = random.sample(
                list(remaining.elements()), state.players[your_index].deckCount
            )
            remaining -= Counter(your_sampled_deck)
            your_sampled_prize = random.sample(
                list(remaining.elements()), len(state.players[your_index].prize)
            )
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
            root, sample = create_node(
                parent=None,
                search_state=search_state,
                your_index=your_index,
                your_deck=your_deck,
                opponents_deck=opp_cards["deck"],
                model=model,
                batched_inference=self.batched_inference,
            )
            if sample_idx == 0:
                final_sample = sample
                root_actions = [tuple(c.select) for c in root.children]
                for action in root_actions:
                    aggregated_visits[action] = 0
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

            # Search
            for _ in range(self.search_count_per_sample):
                current = root
                while True:
                    value = -1e9
                    c = 0.4 * math.sqrt(current.visit)
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
                            next = child

                    if next.node is None:
                        search_state = search_step(current.state.searchId, next.select)
                        next.node, _ = create_node(
                            parent=current,
                            search_state=search_state,
                            your_index=your_index,
                            your_deck=your_deck,
                            opponents_deck=opp_cards["deck"],
                            model=model,
                            batched_inference=self.batched_inference,
                        )
                        break
                    else:
                        current = next.node
                        if current.state.observation.current.result >= 0:
                            current.backprop(current.value)
                            break

            # Generate training data
            sample.value = root.total / root.visit
            visits = [
                child.node.visit if child.node is not None else 0
                for child in root.children
            ]
            sum_visits = sum(visits)
            if sum_visits > 0:
                for i in range(len(root.children)):
                    sample.policy[i] = visits[i] / sum_visits
            else:
                for i in range(len(root.children)):
                    sample.policy[i] = 1.0 / len(root.children)
            for child in root.children:
                if child.node is not None:
                    action_tuple = tuple(child.select)
                    if action_tuple in aggregated_visits:
                        aggregated_visits[action_tuple] += child.node.visit

            if root.visit > 0:
                aggregated_value += root.total / root.visit

            search_end()
        best_action = max(aggregated_visits, key=aggregated_visits.get)
        total_visits = sum(aggregated_visits.values())
        if total_visits > 0:
            for i, action in enumerate(root_actions):
                final_sample.policy[i] = aggregated_visits[action] / total_visits
        else:
            for i in range(len(root_actions)):
                final_sample.policy[i] = 1.0 / len(root_actions)
        # final_sample.value = aggregated_value
        return (list(best_action), final_sample)

    def __call__(self, obs):
        return self.mcts_agent(obs, self.deck, self.model, self.is_eval)

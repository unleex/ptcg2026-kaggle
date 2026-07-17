import torch
import math

from cg.api import (
    SearchState,
)
import transformer_mcts.transformer as transformer
import random

from cg.api import (
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)

SEARCH_COUNT = 50  # MCTS Search count


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
            current_deck = [1072] * obs.current.players[opponent_active_index].deckCount

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


# We will perform exploration using MCTS and select actions. At the same time, we will also generate training data.
def mcts_agent(
    obs_dict: dict, your_deck: list[int], model: transformer.MyModel
) -> tuple[list[int], transformer.LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=random.sample(
            your_deck, state.players[your_index].deckCount
        ),  # Randomly select from deck.
        your_prize=random.sample(
            your_deck, len(state.players[your_index].prize)
        ),  # Randomly select from deck.
        opponent_deck=[1072]
        * state.players[
            1 - your_index
        ].deckCount,  # Fill with Snorlax (There is no deep meaning).
        opponent_prize=[1]
        * len(
            state.players[1 - your_index].prize
        ),  # Fill with Basic Energy (There is no deep meaning)
        opponent_hand=[1]
        * state.players[1 - your_index].handCount,  # Fill with Basic Energy.
        opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
    )  # Fill with Snorlax.
    root, sample = create_node(
        None, search_state, your_index, your_deck, model
    )  # Create root node.
    DIRICHLET_ALPHA = 0.3  # Typical for games with many actions

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
            child.prob * (1 - EXPLORATION_FRACTION) + noise[i] * EXPLORATION_FRACTION
        )
    # Search
    for _ in range(SEARCH_COUNT):
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
                    current, search_state, your_index, your_deck, model
                )
                break
            else:
                current = next.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select the most visited node.
    max_child = None
    max_visit = -1
    min_value = 10
    for child in root.children:
        if child.node is not None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    # Generate training data
    sample.value = root.total / root.visit
    visits = [
        child.node.visit if child.node is not None else 0 for child in root.children
    ]
    sum_visits = sum(visits)
    if sum_visits > 0:
        for i in range(len(root.children)):
            sample.policy[i] = visits[i] / sum_visits
    else:
        for i in range(len(root.children)):
            sample.policy[i] = 1.0 / len(root.children)

    search_end()
    return (max_child.select, sample)

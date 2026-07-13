import math

from src.cg.api import (
    SearchState,
)
import baseline.model as model


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
        if self.parent != None:
            self.parent.backprop(value)


def create_node(
    parent: Node | None,
    search_state: SearchState,
    your_index: int,
    your_deck: list[int],
    model: model.MyModel,
) -> tuple[Node, model.LearnSample | None]:
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

        sv_enc = model.get_encoder_input(obs, your_deck)
        sv_dec = model.get_decoder_input(obs, actions)
        value, policy = model.eval_nn(sv_enc, sv_dec, model)
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
        sample = model.LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)

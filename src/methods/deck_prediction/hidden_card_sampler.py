import random
from collections import Counter

from cg.api import Observation
from methods.deck_prediction.card_tracker import OpponentCardTracker
from utils.utils import is_basic_pokemon


class SimpleSampler:
    def __init__(
        self,
        opponent_deck: list[int],
        epsilon: float = 0.005,
    ):
        self.opponent_deck = opponent_deck
        self.epsilon = epsilon
        self.tracker = OpponentCardTracker()

    def update(self, obs):
        self.tracker.update(obs)

    def reset(self):
        self.tracker = OpponentCardTracker()

    def sample_opponent(self, obs: Observation):
        your_index = obs.current.yourIndex
        state = obs.current
        opp_idx = 1 - your_index
        opp_state = state.players[opp_idx]
        active = opp_state.active

        num_hand = opp_state.handCount
        num_deck = opp_state.deckCount
        num_prize = len(opp_state.prize)
        has_hidden_active = len(active) > 0 and active[0] is None

        public_cards = [card.id for card in opp_state.discard]

        def add_poke(pokemon):
            if pokemon is None:
                return
            public_cards.append(pokemon.id)
            for card in pokemon.energyCards:
                public_cards.append(card.id)
            for card in pokemon.tools:
                public_cards.append(card.id)
            for card in pokemon.preEvolution:
                public_cards.append(card.id)

        # 1. Field (Active & Bench)
        if opp_state.active and opp_state.active[0]:
            add_poke(opp_state.active[0])
        for p in opp_state.bench:
            add_poke(p)

        # 2. Opponent's Stadium (if active)
        for stadium_card in state.stadium:
            if stadium_card.playerIndex == opp_idx:
                public_cards.append(stadium_card.id)

        known_hand_ids = []  # self.tracker.get_known_hand_card_ids()

        possible = list(
            (
                Counter(self.opponent_deck)
                - Counter(public_cards)
                - Counter(known_hand_ids)
            ).elements()
        )
        random.shuffle(possible)

        if has_hidden_active:
            basic_candidates = [c_id for c_id in possible if is_basic_pokemon(c_id)]

            hidden_active_id = basic_candidates[0]
            possible.remove(hidden_active_id)
            hidden_active = [hidden_active_id]
        else:
            hidden_active = []

        unknown_hand_count = num_hand - len(known_hand_ids)
        sampled_unknown_hand = possible[:unknown_hand_count]
        opponent_hand = known_hand_ids + sampled_unknown_hand

        deck = possible[unknown_hand_count : unknown_hand_count + num_deck]
        prize = possible[
            unknown_hand_count + num_deck : unknown_hand_count + num_deck + num_prize
        ]

        return {
            "hand": opponent_hand,
            "deck": deck,
            "prize": prize,
            "hidden_active": hidden_active,
        }

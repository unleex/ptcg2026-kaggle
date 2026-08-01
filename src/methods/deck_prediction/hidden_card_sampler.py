import random
from collections import Counter

from cg.api import Observation
from methods.deck_prediction.card_tracker import OpponentCardTracker
from utils.utils import get_public_cards, is_basic_pokemon


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
        public_cards = get_public_cards(opp_state)
        # 2. Opponent's Stadium (if active)
        stadium_card = state.stadium
        if stadium_card and stadium_card[0].playerIndex == opp_idx:
            public_cards.append(stadium_card.id)

        possible = list(
            (Counter(self.opponent_deck) - Counter(public_cards)).elements()
        )
        random.shuffle(possible)

        if has_hidden_active:
            basic_candidates = [c_id for c_id in possible if is_basic_pokemon(c_id)]

            hidden_active_id = basic_candidates[0]
            possible.remove(hidden_active_id)
            hidden_active = [hidden_active_id]
        else:
            hidden_active = []

        opponent_hand = possible[:num_hand]

        deck = possible[num_hand : num_hand + num_deck]
        prize = possible[num_hand + num_deck : num_hand + num_deck + num_prize]

        return {
            "hand": opponent_hand,
            "deck": deck,
            "prize": prize,
            "hidden_active": hidden_active,
        }

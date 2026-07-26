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

    def _sample_cards(
        self, num_hand, num_deck, num_prize, has_hidden_active, possible: list[int]
    ):
        possible = possible.copy()
        if has_hidden_active:
            # first found basic pokemon
            hidden_active = next(filter(is_basic_pokemon, possible))

            possible.remove(
                hidden_active
            )  # TODO: well i could swap with end and pop in. O(1) but who cares huh
            hidden_active = [hidden_active]
        else:
            hidden_active = None
        opp_cards = {
            "hand": possible[:num_hand],
            "deck": possible[num_hand : num_hand + num_deck],
            "prize": possible[num_hand + num_deck : num_hand + num_deck + num_prize],
            "hidden_active": hidden_active,
        }
        return opp_cards

    def sample_opponent(self, obs: Observation):
        your_index = obs.current.yourIndex
        state = obs.current
        opp_state = state.players[1 - your_index]
        active = opp_state.active
        public_cards = [card.id for card in opp_state.discard]

        def add_poke(pokemon):
            for card in pokemon.energyCards:
                public_cards.append(card.id)
            for card in pokemon.tools:
                public_cards.append(card.id)
            for card in pokemon.preEvolution:
                public_cards.append(card.id)

        # 2. Field (Active & Bench)
        if opp_state.active and opp_state.active[0]:
            add_poke(opp_state.active[0])
        for p in opp_state.bench:
            add_poke(p)
        possible = list(
            (Counter(self.opponent_deck) - Counter(public_cards)).elements()
        )
        random.shuffle(possible)
        return self._sample_cards(
            num_hand=state.players[1 - your_index].handCount,
            num_deck=state.players[1 - your_index].deckCount,
            num_prize=len(state.players[1 - your_index].prize),
            has_hidden_active=len(active) > 0 and active[0] is None,
            possible=possible,
        )

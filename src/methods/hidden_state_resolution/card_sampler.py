import math
import random
from collections import Counter
from dataclasses import dataclass

from cg.api import Observation
from methods.hidden_state_resolution.card_tracker import OpponentCardTracker
from utils.utils import get_public_cards, is_basic_pokemon


class SoftWeightBayesianFilterSampler:
    @dataclass
    class DeckData:
        deck: list[int]
        prob: float = 0.0
        name: str = ""

    def __init__(
        self,
        deck_pool: list[list[int]],
        deck_names: list[str],
        card_role_weights: dict[int, float],
        epsilon: float = 0.005,
        missing_card_score: float = 1e-3,
    ):
        self.decks_data = [
            self.DeckData(deck, 1 / len(deck_pool), name)
            for deck, name in zip(deck_pool, deck_names)
        ]
        self.card_role_weights = card_role_weights
        self.epsilon = epsilon
        self.tracker = OpponentCardTracker()
        self.missing_card_score = missing_card_score

    def update(self, obs):
        self.tracker.update(obs)
        known_cards = self.tracker.get_known_hand_card_ids()
        prior_log_prob = -math.log(len(self.decks_data))

        # Compute raw log scores locally
        log_scores = []
        for deck_data in self.decks_data:
            deck_copy = deck_data.deck.copy()
            log_score = prior_log_prob

            for card_id in known_cards:
                card_importance = self.card_role_weights[card_id]
                if card_id in deck_copy:
                    draw_prob = deck_copy.count(card_id) / len(deck_copy)
                    deck_copy.remove(card_id)
                else:
                    draw_prob = self.missing_card_score

                log_score += math.log(draw_prob) * card_importance

            log_scores.append(log_score)

        # Normalize and write back to DeckData
        max_log = max(log_scores)
        unnormalized = [math.exp(s - max_log) for s in log_scores]
        total = sum(unnormalized)

        for data, unnorm_p in zip(self.decks_data, unnormalized):
            data.prob = unnorm_p / total

    def reset(self):
        self.tracker.reset()

    def sample_deck(self):
        return random.choices(
            [deck_data.deck for deck_data in self.decks_data],
            weights=[deck_data.prob for deck_data in self.decks_data],
            k=1,
        )[0]

    def sample_opponent(self, obs: Observation):
        sampled_deck = self.sample_deck()
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

        possible = list((Counter(sampled_deck) - Counter(public_cards)).elements())
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

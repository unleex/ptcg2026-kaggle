from pathlib import Path
import json


class EloRating:
    """
    Standard ELO rating system.

    Each registered model gets a rating. After every game:
      expected_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
      R_A += K * (score_A - expected_A)   where score: win=1, draw=0.5, loss=0

    K controls how fast ratings move:
      - High K (e.g. 64): fast, good for early calibration with few games
      - Low K  (e.g. 16): stable, good for fine-grained comparison
    """

    def __init__(self, k: float = 32, initial: float = 1000.0):
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}
        self.games_played: dict[str, int] = {}

    def register(self, name: str):
        if name not in self.ratings:
            self.ratings[name] = self.initial
            self.games_played[name] = 0

    def expected(self, name_a: str, name_b: str) -> float:
        """Probability that A beats B according to current ratings."""
        return 1.0 / (1.0 + 10 ** ((self.ratings[name_b] - self.ratings[name_a]) / 400))

    def update(self, name_a: str, name_b: str, a_score: float, custom_k=None):
        """
        Updates ratings using an aggregated win rate (0.0 to 1.0) instead of individual rows.
        a_score: 1.0 if player A won, 0.5 if draw, 0.0 if player B won.
        """
        assert name_a != name_b
        k = custom_k if custom_k is not None else self.k
        # 1. Calculate expected win probabilities based on current frozen ratings
        e_a = self.expected(name_a, name_b)
        e_b = 1.0 - e_a

        # 2. Policy B's empirical win rate is simply the inverse of Policy A's
        score_b = 1.0 - a_score

        # 3. Compute deltas directly using the fractional scores
        delta_a = k * (a_score - e_a)
        delta_b = k * (score_b - e_b)

        # 4. Mutate global states synchronously
        self.ratings[name_a] += delta_a
        self.ratings[name_b] += delta_b

        self.games_played[name_a] += 1
        self.games_played[name_b] += 1

    def summary(self) -> str:
        rows = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        lines = [
            "=" * 56,
            f"{'ELO RANKINGS':^56}",
            "=" * 56,
            f"  {'Model':<36} {'ELO':>6}  {'Games':>5}",
            "-" * 56,
        ]
        for name, rating in rows:
            lines.append(f"  {name:<36} {rating:>6.1f}  {self.games_played[name]:>5}")
        lines.append("=" * 56)
        return "\n".join(lines)

    def save_json(self, path):
        data = {"ratings": self.ratings, "games_played": self.games_played}
        json.dump(data, open(path, "w+"))

    def load_json(self, path):
        data = json.load(open(path, "r"))
        self.ratings = data["ratings"]
        self.games_played = data["games_played"]

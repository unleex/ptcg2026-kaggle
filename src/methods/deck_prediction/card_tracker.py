class OpponentCardTracker:
    def __init__(self):
        # Map: card serial -> card id
        self.seen_cards: dict[int, int] = {}

    def update(self, obs):
        if obs.current is None:
            return

        opp_idx = 1 - obs.current.yourIndex

        def record_card(card_id: int | None, serial: int | None, p_idx: int | None):
            if (p_idx is None or p_idx == opp_idx) and serial and card_id:
                self.seen_cards[serial] = card_id

        def record_card_obj(card):
            if card and card.playerIndex == opp_idx and card.id and card.serial:
                self.seen_cards[card.serial] = card.id

        def record_pokemon(poke):
            if poke is None:
                return
            if poke.id and poke.serial:
                self.seen_cards[poke.serial] = poke.id
            for c in poke.energyCards:
                record_card_obj(c)
            for c in poke.tools:
                record_card_obj(c)
            for c in poke.preEvolution:
                record_card_obj(c)

        opp_state = obs.current.players[opp_idx]

        # 1. Active & Bench
        if opp_state.active and opp_state.active[0]:
            record_pokemon(opp_state.active[0])
        for poke in opp_state.bench:
            record_pokemon(poke)

        # 2. Public Discard & Prize cards
        for card in opp_state.discard:
            record_card_obj(card)
        for card in opp_state.prize:
            record_card_obj(card)
        if opp_state.hand:
            for card in opp_state.hand:
                record_card_obj(card)

        # 3. Stadium & Looking cards
        for card in obs.current.stadium:
            record_card_obj(card)
        if obs.current.looking:
            for card in obs.current.looking:
                record_card_obj(card)

        # 4. Deck Search Cards (obs.select.deck is populated during deck lookups)
        if obs.select:
            if obs.select.deck:
                for card in obs.select.deck:
                    record_card_obj(card)
            for opt in obs.select.option:
                record_card(opt.cardId, opt.serial, opt.playerIndex)

        # 5. Logs (captures card reveals, moves, plays, attaches, evolutions)
        for log in obs.logs:
            p = log.playerIndex
            if p == opp_idx or p is None:
                record_card(log.cardId, log.serial, p)
                record_card(log.cardIdActive, log.serialActive, p)
                record_card(log.cardIdBench, log.serialBench, p)
                record_card(log.cardIdBefore, log.serialBefore, p)
                record_card(log.cardIdAfter, log.serialAfter, p)
                record_card(log.cardIdTarget, log.serialTarget, p)

    def get_visible_card_ids(self) -> list[int]:
        return list(self.seen_cards.values())

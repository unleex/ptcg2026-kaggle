from cg.api import AreaType, LogType, Observation


class OpponentCardTracker:
    def __init__(self):
        self.seen_cards: dict[int, int] = {}
        self.hand_cards: dict[int, int] = {}

    def reset(self):
        self.seen_cards = {}
        self.hand_cards = {}

    def update(self, obs: Observation):
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

        if opp_state.active and opp_state.active[0]:
            record_pokemon(opp_state.active[0])
        for poke in opp_state.bench:
            record_pokemon(poke)

        for card in opp_state.discard:
            record_card_obj(card)
        for card in opp_state.prize:
            record_card_obj(card)

        for card in obs.current.stadium:
            record_card_obj(card)
        if obs.current.looking:
            for card in obs.current.looking:
                record_card_obj(card)

        if obs.select:
            if obs.select.deck:
                for card in obs.select.deck:
                    record_card_obj(card)
            for opt in obs.select.option:
                if opt.playerIndex == opp_idx and opt.cardId and opt.serial:
                    self.seen_cards[opt.serial] = opt.cardId

        for log in obs.logs:
            p = log.playerIndex
            if p == opp_idx or p is None:
                record_card(log.cardId, log.serial, p)
                record_card(log.cardIdActive, log.serialActive, p)
                record_card(log.cardIdBench, log.serialBench, p)
                record_card(log.cardIdBefore, log.serialBefore, p)
                record_card(log.cardIdAfter, log.serialAfter, p)
                record_card(log.cardIdTarget, log.serialTarget, p)

            if p == opp_idx:
                if log.type == LogType.MOVE_CARD:
                    if log.fromArea == AreaType.HAND and log.serial:
                        self.hand_cards.pop(log.serial, None)
                    if log.toArea == AreaType.HAND and log.serial and log.cardId:
                        self.hand_cards[log.serial] = log.cardId

                elif log.type == LogType.MOVE_CARD_REVERSE:
                    if log.fromArea == AreaType.HAND and log.serial:
                        self.hand_cards.pop(log.serial, None)

                elif log.type in (
                    LogType.PLAY,
                    LogType.ATTACH,
                    LogType.EVOLVE,
                    LogType.DEVOLVE,
                ):
                    if log.serial:
                        self.hand_cards.pop(log.serial, None)

        public_serials = set()
        if opp_state.active and opp_state.active[0]:
            public_serials.add(opp_state.active[0].serial)
            for c in (
                opp_state.active[0].energyCards
                + opp_state.active[0].tools
                + opp_state.active[0].preEvolution
            ):
                public_serials.add(c.serial)

        for poke in opp_state.bench:
            public_serials.add(poke.serial)
            for c in poke.energyCards + poke.tools + poke.preEvolution:
                public_serials.add(c.serial)

        for c in opp_state.discard:
            public_serials.add(c.serial)

        for c in obs.current.stadium:
            if c and c.playerIndex == opp_idx:
                public_serials.add(c.serial)

        for serial in public_serials:
            self.hand_cards.pop(serial, None)

        current_hand_count = opp_state.handCount
        if len(self.hand_cards) > current_hand_count:
            excess = len(self.hand_cards) - current_hand_count
            for _ in range(excess):
                self.hand_cards.popitem()

    def get_known_hand_card_ids(self) -> list[int]:
        return list(self.hand_cards.values())

    def get_visible_card_ids(self) -> list[int]:
        return list(self.seen_cards.values())

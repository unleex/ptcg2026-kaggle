from enum import IntEnum

from cg.api import all_card_data, CardData, CardType


class CardKind(IntEnum):
    ITEM = 1
    TOOL = 2  # Pokémon Tool
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6
    BASIC_POKEMON = 7
    STAGE1_POKEMON = 8
    STAGE2_POKEMON = 9


def get_card_kind(card: CardData) -> CardKind:
    if card.cardType == CardType.ITEM:
        return CardKind.ITEM
    elif card.cardType == CardType.TOOL:
        return CardKind.TOOL
    elif card.cardType == CardType.SUPPORTER:
        return CardKind.SUPPORTER
    elif card.cardType == CardType.STADIUM:
        return CardKind.STADIUM
    elif card.cardType == CardType.BASIC_ENERGY:
        return CardKind.BASIC_ENERGY
    elif card.cardType == CardType.SPECIAL_ENERGY:
        return CardKind.SPECIAL_ENERGY
    elif card.cardType == CardType.POKEMON:
        if card.basic:
            return CardKind.BASIC_POKEMON
        elif card.stage1:
            return CardKind.STAGE1_POKEMON
        elif card.stage2:
            return CardKind.STAGE2_POKEMON


BASIC_POKEMON_IDS = {card.cardId for card in all_card_data() if card.basic}
CARD_ID_TO_KIND = {card.cardId: get_card_kind(card) for card in all_card_data()}


def is_basic_pokemon(card_id):
    return card_id in BASIC_POKEMON_IDS


def get_public_cards(player_state):

    public_cards = [card.id for card in player_state.discard]

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
    if player_state.active and player_state.active[0]:
        add_poke(player_state.active[0])
    for p in player_state.bench:
        add_poke(p)
    return public_cards

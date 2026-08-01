from cg.api import all_card_data

BASIC_POKEMON_IDS = {card.cardId for card in all_card_data() if card.basic}


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

from cg.api import all_card_data

BASIC_POKEMON_IDS = {card.cardId for card in all_card_data() if card.basic}


def is_basic_pokemon(card_id):
    return card_id in BASIC_POKEMON_IDS

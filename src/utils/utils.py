import functools
import os
import traceback

from cg.api import all_card_data

BASIC_POKEMON_IDS = {card.cardId for card in all_card_data() if card.basic}


def is_basic_pokemon(card_id):
    return card_id in BASIC_POKEMON_IDS


def catch_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(
                f"\n=== CRASH IN WORKER PID {os.getpid()} ===\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            raise

    return wrapper

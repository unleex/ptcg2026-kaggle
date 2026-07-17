import typing


class Player:
    def __init__(
        self,
        model: typing.Callable,
        name: str,
        deck: list[int],
        is_trainable: bool = True,
    ):
        self.model = model
        self.name = name
        self.deck = deck
        self.is_trainable = is_trainable

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

import json
import random
import wandb
from tqdm import tqdm

import torch
import torch.nn
import torch.optim
from pathlib import Path

from kaggle_ptcg_engine.ptcg.cg.api import (
    SelectContext,
    all_attack,
    all_card_data,
)
from kaggle_ptcg_engine.ptcg.cg.game import (
    battle_start,
    battle_finish,
    battle_select,
    visualize_data,
)
import transformer_mcts.transformer as transformer
from transformer_mcts.mcts import mcts_agent
from elo import EloRating
from rule_based_mega_lucario_ex.agent import (
    agent as rule_based_lucario_agent,
    my_deck as mega_lucario_ex_deck,
)

results_dir = Path("results_working")
results_dir.mkdir(exist_ok=True)
weights_dir = results_dir / Path("out")
weights_dir.mkdir(exist_ok=True)
vis_savedir = results_dir / Path("visuals")
vis_savedir.mkdir(exist_ok=True)

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId: c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1  # Max Card ID + 1

attack_count = (
    max(all_attack(), key=lambda a: a.attackId).attackId + 1
)  # Max Attack ID + 1

num_words_encoder = 24
encoder_size = 22000  # Encoder input size exceeding the vocabulary size

decoder_main_feature = 8  # Feature count of SelectContext.Main
decoder_attack_offset = 14  # First index of Attack feature
decoder_card_offset = (
    decoder_attack_offset + attack_count
)  # First index of Card Feature
decoder_size = (
    decoder_card_offset
    + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count
)  # Decoder input vocabulary size


# Helper class to construct batch inputs for the neural network.
class LearnInput:
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: transformer.SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.MyModel(128, 2, 256, 1, 1)
model = model.to(device)
model_path = weights_dir / "model.pth"
if model_path.exists():
    print("Restoring", model_path)
    model.load_state_dict(torch.load(model_path, weights_only=False))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function

elo = EloRating(initial=600)
elo_data_path = results_dir / "elo.json"
if elo_data_path.exists():
    print("Restoring elo from", elo_data_path)
    elo.load_json(elo_data_path)


def play_and_collect_samples(
    player1, deck1, player2, deck2, player1_is_trainable, player2_is_trainable
):
    """
    Play one game and return obtained LearnSamples when needed
    -----
    Returns
    samples: list of length 2: samples for player 1, and samples for player 2. If some player is not trainable, their samples will be empty
    obs: last observation that concludes the game
    """
    obs_log = [""]
    action_log = [None]
    obs, start_data = battle_start(deck1, deck2)
    if start_data.errorPlayer >= 0:
        error = "Deck error."
        if start_data.errorType == 1:
            error = "The deck contains invalid card ID."
        elif start_data.errorType == 2:
            error = "You can include up to four cards with the same name in the deck, excluding basic Energy cards."
        elif start_data.errorType == 3:
            error = "There are no Basic Pokémon in the deck."
        elif start_data.errorType == 4:
            error = "You can include only one Ace Spec card in the deck."
        raise ValueError(error)
    samples: list[list[transformer.LearnSample]] = [
        [],
        [],
    ]  # [Player0 samples, Player1 samples]
    while True:
        if obs["current"]["result"] >= 0:
            break
        your_index = obs["current"]["yourIndex"]
        if your_index == 0:
            # We play as index 0, generate MCTS actions and training samples
            selected, sample = player1(obs)
            if player1_is_trainable:
                samples[obs["current"]["yourIndex"]].append(sample)

        else:
            selected, sample = player2(obs)
            if player2_is_trainable:
                samples[obs["current"]["yourIndex"]].append(sample)

        obs_log.append(obs)
        action_log.append(selected)
        obs = battle_select(selected)

    # For visualiation
    vis = json.loads(visualize_data())
    for i in range(len(vis)):
        vis[i]["obs"] = obs_log[i]
        vis[i]["action"] = [action_log[i], action_log[i]]
    with open(vis_savedir / f"vis{len(list(vis_savedir.iterdir()))}.json", "w") as file:
        json.dump(vis, file)
    battle_finish()  # Finalize the game.
    return samples, action_log, obs_log, obs


# A sample deck for training.
snowy_deck = [
    721,
    721,
    722,
    722,
    722,
    722,
    723,
    723,
    723,
    723,
    1092,
    1121,
    1121,
    1145,
    1145,
    1163,
    1163,
    1219,
    1219,
    1219,
    1219,
    1227,
    1227,
    1227,
    1227,
    1262,
    1262,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
]
player1_deck = mega_lucario_ex_deck


def player1(obs):
    return mcts_agent(obs, player1_deck, model)


player1_name = "model"
player1_is_trainable = True
elo.register(player1_name)


def select_player2_val():

    player2_name = "rule_based_lucario"

    # Trainables return second item as LearnSample, this does not.
    def player2(*args, **kwargs):
        return (rule_based_lucario_agent(*args, **kwargs), None)

    player2_deck = mega_lucario_ex_deck
    player2_is_trainable = False

    return player2_name, player2, player2_deck, player2_is_trainable


def select_player2_train():
    if random.randint(1, 100) < 0:
        # Same model, but trains to play another deck
        model = transformer.MyModel(128, 2, 256, 1, 1)
        if model_path.exists():
            model.load_state_dict(
                torch.load(open(model_path, mode="rb"), weights_only=False)
            )

        def player2(obs):
            return mcts_agent(obs, mega_lucario_ex_deck, model)

        player2_deck = mega_lucario_ex_deck
        player2_is_trainable = True
        player2_name = player1_name

    else:
        player2_name = "rule_based_lucario"

        # Trainables return second item as LearnSample, this does not.
        def player2(*args, **kwargs):
            return (rule_based_lucario_agent(*args, **kwargs), None)

        player2_deck = mega_lucario_ex_deck
        player2_is_trainable = False

    return player2_name, player2, player2_deck, player2_is_trainable


# The main training loop.
if __name__ == "__main__":
    wandb.init(project="ptcg-rl", name="transformer-mcts-training")

    for counter in range(50):
        elo.register(str(player1_name))
        sample_list: list[
            transformer.LearnSample
        ] = []  # List of training data samples.

        model.eval()
        with torch.inference_mode():
            # Evaluation
            results = [0, 0, 0]

            for i in tqdm(range(50), desc=f"Evaluating Epoch {counter}..."):
                player2_name, player2, player2_deck, player2_is_trainable = (
                    select_player2_val()
                )
                elo.register(player2_name)

                _, action_log, obs_log, game_result = play_and_collect_samples(
                    player1=player1,
                    player2=player2,
                    deck1=player1_deck,
                    deck2=player2_deck,
                    player1_is_trainable=player1_is_trainable,
                    player2_is_trainable=player2_is_trainable,
                )

                if game_result["current"]["result"] == 2:  # Draw
                    elo_our_score = 0.5
                    results[2] += 1
                elif game_result["current"]["result"] == 0:  # Win
                    elo_our_score = 1
                    results[0] += 1
                else:  # Lose
                    elo_our_score = 0
                    results[1] += 1
                elo.update(
                    name_a=str(player1_name),
                    name_b=str(player2_name),
                    a_score=elo_our_score,
                )

            win_rate = (
                100 * results[0] // (results[0] + results[1])
                if (results[0] + results[1]) > 0
                else 0
            )
            print(f"Evaluation win rate {win_rate}%", flush=True)
            print(elo.summary())
            elo.save_json(elo_data_path)

            # Self Play
            for i in tqdm(range(100), desc=f"Data Collecting Epoch {counter}..."):
                player2_name, player2, player2_deck, player2_is_trainable = (
                    select_player2_train()
                )

                samples, _, _, game_result = play_and_collect_samples(
                    player1=player1,
                    player2=player2,
                    deck1=player1_deck,
                    deck2=player2_deck,
                    player1_is_trainable=player1_is_trainable,
                    player2_is_trainable=player2_is_trainable,
                )
                # Calculate the training labels and add them to the training data list.
                for i in range(2):
                    LAMBDA = 0.9
                    # The final value is 1.0 for a win and -1.0 for a loss.
                    value = 1.0 if i == game_result["current"]["result"] else -1.0

                    # Iterate backwards from the end of the game to calculate values.
                    for sample in reversed(samples[i]):
                        label = (value + sample.value) * 0.5
                        value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                        sample.value = label
                        sample_list.append(sample)

        # Train on the training data collected through self-play.
        print("Training Start.")
        model.train()
        random.shuffle(sample_list)
        BATCH_SIZE = 128
        batch_count = len(sample_list) // BATCH_SIZE

        epoch_loss_enc = 0.0
        epoch_loss_dec = 0.0

        for i in tqdm(range(batch_count), desc=f"Training Epoch {counter}..."):
            # Prepare a batch of data.
            input_enc = LearnInput()
            input_dec = LearnInput()
            mask = []
            label_enc = []
            label_dec = []
            start = BATCH_SIZE * i
            for j in range(start, start + BATCH_SIZE):
                sample = sample_list[j]
                input_enc.add(sample.sv_enc)
                input_dec.add(sample.sv_dec)
                label_enc.append(sample.value)
                label_dec.extend(sample.policy)
                for _ in range(len(sample.policy)):
                    mask.append(1.0)
                for _ in range(64 - len(sample.policy)):
                    mask.append(0.0)
                    label_dec.append(0.0)
                    input_dec.offset.append(len(input_dec.index))

            # Convert data to PyTorch tensors.
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
            mask_tensor = mask_tensor.view(BATCH_SIZE, -1)
            label_tensor_enc = torch.tensor(
                label_enc, dtype=torch.float32, device=device
            )
            label_tensor_enc = label_tensor_enc.view(BATCH_SIZE, -1)
            label_tensor_dec = torch.tensor(
                label_dec, dtype=torch.float32, device=device
            )
            label_tensor_dec = label_tensor_dec.view(BATCH_SIZE, -1)

            optimizer.zero_grad()

            # Get model predictions for the batch.
            out_enc, out_dec = model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
            )

            # Calculate loss.
            loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
            loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
            loss_dec = loss_dec * mask_tensor
            loss_dec = loss_dec.sum() / float(BATCH_SIZE)
            loss = loss_enc + loss_dec

            epoch_loss_enc += loss_enc.item()
            epoch_loss_dec += loss_dec.item()

            # Backpropagate the loss and update model parameters.
            loss.backward()
            optimizer.step()

        avg_loss_enc = epoch_loss_enc / batch_count if batch_count > 0 else 0
        avg_loss_dec = epoch_loss_dec / batch_count if batch_count > 0 else 0

        # Safely extract the ELO score from the EloRating dictionary if accessible
        current_elo = getattr(elo, "rating_dict", getattr(elo, "ratings", {})).get(
            str(player1_name), 600
        )

        wandb.log(
            {
                "epoch": counter,
                "eval_win_rate": win_rate,
                "elo": current_elo,
                "loss_encoder": avg_loss_enc,
                "loss_decoder": avg_loss_dec,
                "loss_total": avg_loss_enc + avg_loss_dec,
            }
        )
        torch.save(model.state_dict(), model_path)
        print("Training Finish.")

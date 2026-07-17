import json
import random
import wandb
from tqdm import tqdm

import torch
import torch.nn
import torch.optim
from pathlib import Path

from cg.api import (
    SelectContext,
    all_attack,
    all_card_data,
)
from cg.game import battle_start, battle_finish, battle_select, visualize_data
import transformer_mcts.transformer as transformer
from transformer_mcts.mcts import mcts_agent
from elo import EloRating
from rule_based_mega_lucario_ex.agent import (
    agent as rule_based_lucario_agent,
    my_deck as mega_lucario_ex_deck,
)

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


# A sample deck for training.
sample_deck = [
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.MyModel(128, 2, 256, 1, 1)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
weights_dir = results_dir / Path("out")
weights_dir.mkdir(exist_ok=True)
vis_savedir = results_dir / Path("visuals")
vis_savedir.mkdir(exist_ok=True)


elo = EloRating(initial=600)
elo_data_path = results_dir / "elo.json"
if elo_data_path.exists():
    print("Restoring elo from", elo_data_path)
    elo.load_json(elo_data_path)


def _selected_to_action(selected):
    if isinstance(selected, list):
        return selected[0]
    return int(selected)


def _batch_row(value, index):
    if isinstance(value, dict):
        return {k: v[index] for k, v in value.items()}
    return value[index]


def play_and_collect_samples(
    player1,
    deck1,
    player2,
    deck2,
    player1_is_trainable,
    player2_is_trainable,
    batch_size=8,
):
    """
    Play a batch of games and return obtained LearnSamples when needed
    -----
    Returns
    samples: list of length batch_size: samples for each game. If some player is not trainable, their samples will be empty
    obs: last observation that concludes the game
    """

    import numpy as np
    import ptcg_engine as E

    env = E.VectorEnv(deck1, deck2, batch_size, random.randrange(1, 1 << 30))
    obs, _, player, result = env.observe_features()
    obs_log = [[_batch_row(obs, i)] for i in range(batch_size)]
    action_log = [[None] for _ in range(batch_size)]

    samples: list[list[list[transformer.LearnSample]]] = [
        [[], []] for _ in range(batch_size)
    ]  # [Player0 samples, Player1 samples]
    finished = [False for _ in range(batch_size)]
    while True:
        if all(finished):
            break
        actions = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            if finished[i]:
                continue
            if int(result[i]) >= 0:
                finished[i] = True
                continue
            your_index = int(player[i])
            if your_index == 0:
                # We play as index 0, generate MCTS actions and training samples
                selected, sample = player1(_batch_row(obs, i))
                if player1_is_trainable and sample is not None:
                    samples[i][0].append(sample)

            else:
                selected, sample = player2(_batch_row(obs, i))
                if player2_is_trainable and sample is not None:
                    samples[i][1].append(sample)

            obs_log[i].append(_batch_row(obs, i))
            action_log[i].append(selected)
            actions[i] = _selected_to_action(selected)
        obs, reward, done, mask, player, result = env.step_features(actions)
        for i in range(batch_size):
            if int(done[i]) or int(result[i]) >= 0:
                finished[i] = True
    return samples, action_log, obs_log, obs, result


def player1(obs):
    return mcts_agent(obs, sample_deck, model)


player1_is_trainable = True

# The main training loop.

if __name__ == "__main__":
    wandb.init(project="ptcg-rl", name="transformer-mcts-training")

    for counter in range(50):
        current_model_name = weights_dir / ("model" + str(counter) + ".pth")
        if current_model_name.exists():
            print("Restoring ", current_model_name)
            model.load_state_dict(torch.load(open(current_model_name, mode="rb")))
        torch.save(model.state_dict(), current_model_name)  # Save the current model.
        elo.register(str(current_model_name))
        sample_list: list[
            transformer.LearnSample
        ] = []  # List of training data samples.

        model.eval()
        with torch.inference_mode():
            # Evaluation
            results = [0, 0, 0]

            for i in tqdm(range(0, 50, 8), desc=f"Evaluating Epoch {counter}..."):
                current_batch = min(8, 50 - i)
                if i % 2 == 0:
                    player2_path = str(random.choice(list(weights_dir.iterdir())))
                    player2_model = transformer.MyModel(128, 2, 256, 1, 1)
                    player2_model.load_state_dict(
                        torch.load(open(player2_path, mode="rb"), weights_only=False)
                    )
                    player2_name = player2_path

                    def player2(obs):
                        return mcts_agent(obs, sample_deck, player2_model)

                    player2_deck = sample_deck
                    player2_is_trainable = True
                else:
                    player2_name = "rule_based_lucario"

                    # Trainables return second item as LearnSample, this does not.
                    def player2(*args, **kwargs):
                        return (rule_based_lucario_agent(*args, **kwargs), None)

                    player2_deck = mega_lucario_ex_deck
                    player2_is_trainable = False
                elo.register(player2_name)

                _, action_log, obs_log, game_result, result = play_and_collect_samples(
                    player1=player1,
                    player2=player2,
                    deck1=sample_deck,
                    deck2=player2_deck,
                    player1_is_trainable=player1_is_trainable,
                    player2_is_trainable=player2_is_trainable,
                    batch_size=current_batch,
                )

                for j in range(current_batch):
                    if int(game_result[j]["current"]["result"]) == 2:  # Draw
                        elo_our_score = 0.5
                        results[2] += 1
                    elif int(game_result[j]["current"]["result"]) == 0:  # Win
                        elo_our_score = 1
                        results[0] += 1
                    else:  # Lose
                        elo_our_score = 0
                        results[1] += 1
                    elo.update(
                        name_a=str(current_model_name),
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
            for i in tqdm(range(0, 100, 8), desc=f"Data Collecting Epoch {counter}..."):
                current_batch = min(8, 100 - i)
                if i % 2 == 0:
                    player2_path = str(random.choice(list(weights_dir.iterdir())))
                    player2_model = transformer.MyModel(128, 2, 256, 1, 1)
                    player2_model.load_state_dict(
                        torch.load(open(player2_path, mode="rb"), weights_only=False)
                    )
                    player2_name = player2_path

                    def player2(obs):
                        return mcts_agent(obs, sample_deck, player2_model)

                    player2_deck = sample_deck
                    player2_is_trainable = True
                else:
                    player2_name = "rule_based_lucario"

                    # Trainables return second item as LearnSample, this does not.
                    def player2(*args, **kwargs):
                        return (rule_based_lucario_agent(*args, **kwargs), None)

                    player2_deck = mega_lucario_ex_deck
                    player2_is_trainable = False

                samples, _, _, game_result, result = play_and_collect_samples(
                    player1=player1,
                    player2=player2,
                    deck1=sample_deck,
                    deck2=player2_deck,
                    player1_is_trainable=player1_is_trainable,
                    player2_is_trainable=player2_is_trainable,
                    batch_size=current_batch,
                )
                # Calculate the training labels and add them to the training data list.
                for k in range(current_batch):
                    for j in range(2):
                        LAMBDA = 0.9
                        # The final value is 1.0 for a win and -1.0 for a loss.
                        value = (
                            1.0 if j == game_result[k]["current"]["result"] else -1.0
                        )

                        # Iterate backwards from the end of the game to calculate values.
                        for sample in reversed(samples[k][j]):
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
            str(current_model_name), 600
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

        print("Training Finish.")

import time
import torch.multiprocessing as mp
from player import Player
import json
import random
import wandb
from tqdm import tqdm

import torch
import torch.nn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from agents.imitator import ImitationModel
from transformer_mcts.mcts import mcts_agent
from elo import EloRating
from agents.rule_based_lucario import (
    agent as rule_based_lucario_agent,
    my_deck as mega_lucario_ex_deck,
)

# --- Configuration Constants ---
RUN_NAME = "expert"
PRETRAIN_WEIGHTS_PATH = Path("results/imitated.pt")
# Number of epochs to run imitation learning before switching to MCTS
IMITATION_EPOCHS = 0
SELF_PLAY_CLONE_UPDATE_WINRATE_THRESH = 55
BATCH_SIZE = 128
ENTROPY_COEF = 0.02
VALUE_LOSS_WEIGHT = 10
TOTAL_EPOCHS = 500
TRAIN_ITERATIONS = 100
VAL_ITERATIONS = 50


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


def play_and_collect_samples(player1: Player, player2: Player):
    """Play one game and return obtained LearnSamples when needed."""
    if hasattr(player1.model, "reset"):
        player1.model.reset()
    if hasattr(player2.model, "reset"):
        player2.model.reset()

    obs_log = [""]
    action_log = [None]
    obs, start_data = battle_start(player1.deck, player2.deck)
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
            if player1.is_trainable and sample is not None:
                samples[0].append(sample)
        else:
            selected, sample = player2(obs)
            if player2.is_trainable and sample is not None:
                samples[1].append(sample)

        obs_log.append(obs)
        action_log.append(selected)
        obs = battle_select(selected)

    # Compile trajectory sequences if running an imitation agent wrapper
    if hasattr(player1.model, "post_process_samples") and player1.is_trainable:
        samples[0] = player1.model.post_process_samples(obs["current"]["result"])
    if hasattr(player2.model, "post_process_samples") and player2.is_trainable:
        samples[1] = player2.model.post_process_samples(obs["current"]["result"])

    vis = json.loads(visualize_data())
    for i in range(len(vis)):
        vis[i]["obs"] = obs_log[i]
        vis[i]["action"] = [action_log[i], action_log[i]]
    battle_finish()
    return samples, action_log, obs_log, obs, vis


# Environment setup structures
results_dir = Path("results") / RUN_NAME
results_dir.mkdir(exist_ok=True, parents=True)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.MyModel(
    d_model=256,
    num_heads=8,
    d_feedforward=1024,
    num_layers_encoder=4,
    num_layers_decoder=4,
).to(device)

model2 = transformer.MyModel(
    d_model=256,
    num_heads=8,
    d_feedforward=1024,
    num_layers_encoder=4,
    num_layers_decoder=4,
).to(device)
model_path = weights_dir / "model.pth"
model2_path = weights_dir / "model2.pth"
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)

elo = EloRating(initial=600)
elo_data_path = results_dir / "elo.json"


# ----- Players definition -----
class MCTSAgentWrapper:
    def __init__(
        self, model: transformer.MyModel, deck: list[int], is_eval: bool = False
    ):
        self.model = model
        self.deck = deck
        self.is_eval = is_eval

    def __call__(self, obs):
        return mcts_agent(obs, self.deck, self.model, is_eval=self.is_eval)


player1_deck = mega_lucario_ex_deck

player1_mcts = MCTSAgentWrapper(model, player1_deck, is_eval=False)
player2_mcts = MCTSAgentWrapper(model2, player1_deck, is_eval=False)

player1 = Player(model=player1_mcts, name="model", deck=player1_deck, is_trainable=True)
player2 = Player(
    model=player2_mcts, name="model2", deck=player1_deck, is_trainable=False
)


def player_lucario_agent(*args, **kwargs):
    return (rule_based_lucario_agent(*args, **kwargs), None)


player_lucario = Player(
    model=player_lucario_agent,
    name="rule_based_lucario",
    deck=mega_lucario_ex_deck,
    is_trainable=False,
)

if __name__ == "__main__":
    if model_path.exists():
        print("❇️Restoring", model_path)
        model.load_state_dict(torch.load(model_path, weights_only=False))
    elif PRETRAIN_WEIGHTS_PATH.exists():
        print("❇️Restoring", model_path)
        model.load_state_dict(torch.load(PRETRAIN_WEIGHTS_PATH, weights_only=False))

    if model2_path.exists():
        print("❇️Restoring", model2_path)
        model2.load_state_dict(torch.load(model2_path, weights_only=False))
    if elo_data_path.exists():
        print("❇️Restoring elo from", elo_data_path)
        elo.load_json(elo_data_path)

    ctx = mp.get_context("spawn")
    model.share_memory()
    pool = ctx.Pool(processes=50)
    wandb.init(project="ptcg-rl", name=RUN_NAME, resume="allow")
    if (results_dir / "epoch.txt").exists():
        with open(results_dir / "epoch.txt", "r") as f:
            start_epoch = int(f.read())
    else:
        start_epoch = 0
    for _ in range(start_epoch):
        lr_scheduler.step()
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        is_imitating = epoch < IMITATION_EPOCHS
        sample_list: list[transformer.LearnSample] = []

        if is_imitating:
            print(f"--- Epoch {epoch}: IMITATION LEARNING PHASE ---", flush=True)
            p1_active_model = ImitationModel(
                rule_based_lucario_agent, player1_deck, epsilon=0
            )
            p2_active_model = ImitationModel(
                rule_based_lucario_agent, player1_deck, epsilon=0
            )

            player1.model = p1_active_model
            player1.is_trainable = True

            pretrain_partner = Player(
                model=p2_active_model,
                name="model_clone_pretrain",
                deck=player1_deck,
                is_trainable=True,
            )
            val_partner = player_lucario
        else:
            if epoch == IMITATION_EPOCHS:
                torch.save(model.state_dict(), weights_dir / "imitated.pt")
                lr_scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

            print(f"--- Epoch {epoch}: MCTS SELF-PLAY PHASE ---", flush=True)
            player1.model = player1_mcts
            player1.is_trainable = True

            player2.model = player2_mcts
            player2.is_trainable = False

            val_partner = player_lucario

        model.eval()
        with torch.inference_mode():
            # Evaluation Phase
            player1_mcts.is_eval = True
            player2_mcts.is_eval = True

            results = [0, 0, 0]
            start_time = time.perf_counter()
            total_samples = 0

            async_eval_results = []
            for _ in range(VAL_ITERATIONS):
                elo.register(player1.name)
                elo.register(val_partner.name)
                async_eval_results.append(
                    pool.apply_async(
                        play_and_collect_samples, args=(player1, val_partner)
                    )
                )

            pbar = tqdm(async_eval_results, desc=f"Evaluating Epoch {epoch}...")
            for async_res in pbar:
                samples, action_log, obs_log, game_result, vis = async_res.get()

                file_idx = len(list(vis_savedir.iterdir()))
                with open(
                    vis_savedir / f"vis_val_{epoch}_{file_idx}.json", "w"
                ) as file:
                    json.dump(vis, file)

                if game_result["current"]["result"] == 2:  # Draw
                    elo_our_score = 0.5
                    results[2] += 1
                elif game_result["current"]["result"] == 0:  # Win
                    elo_our_score = 1
                    results[0] += 1
                else:  # Loss
                    elo_our_score = 0
                    results[1] += 1
                elo.update(
                    name_a=str(player1.name),
                    name_b=str(val_partner.name),
                    a_score=elo_our_score,
                )

                game_samples = sum(len(player_samples) for player_samples in samples)
                total_samples += game_samples
                elapsed = time.perf_counter() - start_time
                sps = total_samples / elapsed if elapsed > 0 else 0
                pbar.set_postfix(SPS=f"{sps:.1f}", Samples=total_samples)

            win_rate = (
                100 * results[0] // (results[0] + results[1])
                if (results[0] + results[1]) > 0
                else 0
            )
            print(f"Evaluation win rate {win_rate}%", flush=True)
            print(elo.summary())
            elo.save_json(elo_data_path)

            # Data Generation / Training Sampling Phase
            player1_mcts.is_eval = False
            player2_mcts.is_eval = False
            results_train = [0, 0, 0]
            async_train_results = []
            for _ in range(TRAIN_ITERATIONS):
                if is_imitating:
                    train_partner = pretrain_partner
                else:
                    train_partner = player_lucario

                elo.register(player1.name)
                elo.register(train_partner.name)

                if random.random() < 0.5:
                    async_train_results.append(
                        pool.apply_async(
                            play_and_collect_samples, args=(player1, train_partner)
                        )
                    )
                else:
                    async_train_results.append(
                        pool.apply_async(
                            play_and_collect_samples, args=(train_partner, player1)
                        )
                    )

            for async_res in tqdm(
                async_train_results, desc=f"Data Collecting Epoch {epoch}..."
            ):
                samples, _, _, game_result, vis = async_res.get()

                with open(
                    vis_savedir
                    / f"vis_train_{epoch}_{len(list(vis_savedir.iterdir()))}.json",
                    "w",
                ) as file:
                    json.dump(vis, file)

                if game_result["current"]["result"] == 2:
                    results_train[2] += 1
                elif game_result["current"]["result"] == 0:
                    results_train[0] += 1
                else:
                    results_train[1] += 1

                if not is_imitating:
                    for i in range(2):
                        LAMBDA = 0.95
                        if game_result["current"]["result"] == 2:
                            final_outcome = 0.0
                        else:
                            final_outcome = (
                                1.0 if i == game_result["current"]["result"] else -1.0
                            )

                        current_value = final_outcome
                        for sample in reversed(samples[i]):
                            sample.value = current_value
                            sample_list.append(sample)
                            current_value *= LAMBDA
                else:
                    sample_list.extend(samples[0])
                    sample_list.extend(samples[1])

        win_rate_train = (
            100 * results_train[0] // (results_train[0] + results_train[1])
            if (results_train[0] + results_train[1]) > 0
            else 0
        )
        print(f"Train win rate: {win_rate_train}%", flush=True)

        if is_imitating and win_rate >= 55:
            print(
                "Competence threshold reached. Switching to MCTS Self-Play next epoch!",
                flush=True,
            )
            is_imitating = False
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - epoch)
        elif not is_imitating and win_rate < 40:
            print(
                "Model collapsed. Reverting to Imitation Learning next epoch!",
                flush=True,
            )
            is_imitating = True
        if not is_imitating and win_rate_train > SELF_PLAY_CLONE_UPDATE_WINRATE_THRESH:
            print("Updating checkpoint clone target model...", flush=True)
            model2.load_state_dict(model.state_dict())
            torch.save(model2.state_dict(), model2_path)

        # Gradient Optimization Step Phase
        model.train()
        random.shuffle(sample_list)
        batch_count = len(sample_list) // BATCH_SIZE

        epoch_loss_enc = 0.0
        epoch_loss_dec = 0.0
        entropy_epoch = 0.0
        running_expl_var = 0.0
        running_kl_div = 0.0

        for i in tqdm(range(batch_count), desc=f"Training Epoch {epoch}..."):
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

            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(
                BATCH_SIZE, -1
            )
            label_tensor_enc = torch.tensor(
                label_enc, dtype=torch.float32, device=device
            ).view(BATCH_SIZE, -1)
            label_tensor_dec = torch.tensor(
                label_dec, dtype=torch.float32, device=device
            ).view(BATCH_SIZE, -1)

            optimizer.zero_grad()

            out_enc, out_dec = model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
            )

            loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
            masked_logits = out_dec.masked_fill(mask_tensor == 0.0, -1e9)
            loss_dec = torch.nn.functional.cross_entropy(
                masked_logits, label_tensor_dec, reduction="none"
            ).mean()

            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            kl_div = torch.nn.functional.kl_div(
                log_probs, label_tensor_dec, reduction="batchmean"
            )
            probs = torch.exp(log_probs)
            entropy = (
                -(probs * log_probs)
                .masked_fill(mask_tensor == 0.0, 0.0)
                .sum(dim=-1)
                .mean()
            )

            loss = (loss_enc * VALUE_LOSS_WEIGHT) + loss_dec - (ENTROPY_COEF * entropy)

            epoch_loss_enc += loss_enc.item()
            epoch_loss_dec += loss_dec.item()
            entropy_epoch += entropy.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            target_var = torch.var(label_tensor_enc)
            if target_var > 0:
                explained_variance = (
                    1 - torch.var(label_tensor_enc - out_enc) / target_var
                )
            else:
                explained_variance = torch.tensor(float("nan"))
            running_expl_var += explained_variance.item()
            running_kl_div += kl_div.item()

        avg_loss_enc = epoch_loss_enc / batch_count
        avg_loss_dec = epoch_loss_dec / batch_count
        current_elo = elo.ratings[player1.name]

        wandb.log(
            {
                "eval_win_rate": win_rate,
                "train_win_rate": win_rate_train,
                "elo": current_elo,
                "loss_encoder": avg_loss_enc,
                "loss_decoder": avg_loss_dec,
                "entropy": entropy_epoch / batch_count,
                "loss_total": avg_loss_enc + avg_loss_dec,
                "kl_div": running_kl_div / batch_count,
                "expl_var": running_expl_var / batch_count,
                "pretraining_phase": int(is_imitating),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        torch.save(model.state_dict(), model_path)
        with open(results_dir / "epoch.txt", "w+") as f:
            f.write(str(epoch))
        lr_scheduler.step()

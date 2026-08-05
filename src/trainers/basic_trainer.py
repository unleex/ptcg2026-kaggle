import json
import random
import time
from pathlib import Path

import ray
import torch
import torch.multiprocessing as mp
import torch.nn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import decks
import wandb
from agents.rule_based_lucario import (
    agent as rule_based_lucario_agent,
)
from agents.rule_based_lucario import (
    my_deck as mega_lucario_ex_deck,
)
from cg.api import to_observation_class
from elo import EloRating
from inference.inference_server import create_handle
from kaggle_ptcg_engine.ptcg.cg.game import (
    battle_finish,
    battle_select,
    battle_start,
    visualize_data,
)
from methods.hidden_state_resolution.card_sampler import SoftWeightBayesianFilterSampler
from player import Player
from transformer_mcts import transformer
from transformer_mcts.ismcts import ISMCTSPlayer
from utils.utils import CARD_ID_TO_KIND, CardKind

# --- Configuration Constants ---
RUN_NAME = "simple_trainer"
PRETRAIN_WEIGHTS_PATH = Path("results/imitated.pt")
SELF_PLAY_CLONE_UPDATE_WINRATE_THRESH = 55
BATCH_SIZE = 128
ENTROPY_COEF = 0.02
TOTAL_EPOCHS = 500
TRAIN_ITERATIONS = 100
SAMPLE_EPOCHS = 10  # Number of SGD passes over collected MCTS dataset per iteration
LAMBDA_DISCOUNT = 0.95


class LearnInput:
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


@ray.remote
def play_and_collect_samples(player1: Player, player2: Player):
    """Play one game and return obtained LearnSamples."""
    if hasattr(player1.model, "reset"):
        player1.model.reset()
    if hasattr(player2.model, "reset"):
        player2.model.reset()

    obs_log = [""]
    action_log = [None]
    obs, start_data = battle_start(player1.deck, player2.deck)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"Deck error: {start_data.errorType}")

    samples: list[list[transformer.LearnSample]] = [[], []]
    while True:
        if obs["current"]["result"] >= 0:
            print("done")
            break
        your_index = obs["current"]["yourIndex"]
        if your_index == 0:
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

    if hasattr(player1.model, "post_process_samples") and player1.is_trainable:
        samples[0] = player1.model.post_process_samples(obs["current"]["result"])
    if hasattr(player2.model, "post_process_samples") and player2.is_trainable:
        samples[1] = player2.model.post_process_samples(obs["current"]["result"])

    vis = json.loads(visualize_data())
    for i in range(len(vis)):
        vis[i]["obs"] = obs_log[i]
        vis[i]["action"] = [action_log[i], action_log[i]]
    battle_finish()
    return samples, action_log, obs_log, to_observation_class(obs), vis


def player_lucario_agent(*args, **kwargs):
    return (rule_based_lucario_agent(*args, **kwargs), None)


def generate_samples(
    pool: mp.Pool,
    player1: ISMCTSPlayer,
    partner_player: Player,
    num_games: int,
    vis_savedir: Path,
    epoch: int,
    elo: EloRating,
) -> tuple[list[transformer.LearnSample], float]:
    """Runs parallel games and extracts discounted returns into a flat dataset."""
    player1.is_trainable = True
    player1.is_eval = False

    results_train = [0, 0, 0]  # [p1_wins, partner_wins, draws]
    async_results = []

    for _ in range(num_games):
        elo.register(player1.name)
        elo.register(partner_player.name)

        player1_goes_first = random.random() < 0.5
        task_ref = play_and_collect_samples.remote(
            player1 if player1_goes_first else partner_player,
            partner_player if player1_goes_first else player1,
        )
        async_results.append((task_ref, player1_goes_first))

    sample_list: list[transformer.LearnSample] = []
    total_samples = 0
    start_time = time.perf_counter()
    pbar = tqdm(async_results, desc=f"Collecting Data Epoch {epoch}...")
    for task_ref, player1_goes_first in pbar:
        samples, _, _, game_result, vis = ray.get(task_ref)

        with open(
            vis_savedir / f"vis_train_{epoch}_{len(list(vis_savedir.iterdir()))}.json",
            "w",
        ) as file:
            json.dump(vis, file)

        p1_won = (
            game_result.current.result == 0
            if player1_goes_first
            else game_result.current.result == 1
        )

        if game_result.current.result == 2:
            results_train[2] += 1
            elo_score = 0.5
        elif p1_won:
            results_train[0] += 1
            elo_score = 1.0
        else:
            results_train[1] += 1
            elo_score = 0.0

        elo.update(
            name_a=str(player1.name),
            name_b=str(partner_player.name),
            a_score=elo_score if player1_goes_first else 1.0 - elo_score,
        )

        # Value return assignment
        for i in range(2):
            if game_result.current.result == 2:
                final_outcome = 0.0
            else:
                final_outcome = 1.0 if i == game_result.current.result else -1.0

            current_value = final_outcome
            total_samples += len(samples[i])
            for sample in reversed(samples[i]):
                sample.value = current_value
                sample_list.append(sample)
                current_value *= LAMBDA_DISCOUNT
        elapsed = time.perf_counter() - start_time
        sps = total_samples / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix(SPS=f"{sps:.1f}", Samples=total_samples)

    win_rate = (
        100 * results_train[0] // (results_train[0] + results_train[1])
        if (results_train[0] + results_train[1]) > 0
        else 0
    )
    return sample_list, win_rate, sps


def train_on_samples(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: list[transformer.LearnSample],
    batch_size: int,
    sample_epochs: int,
    device: torch.device,
    loss_fn_enc: torch.nn.Module,
    loss_fn_dec: torch.nn.Module,
    entropy_coef: float,
) -> dict[str, float]:
    """Optimizes the neural network parameters over collected samples for multiple epochs."""
    model.train()
    batch_count = len(samples) // batch_size
    if batch_count == 0:
        return {}

    total_loss_enc, total_loss_dec = 0.0, 0.0
    total_entropy, total_kl_div, total_expl_var = 0.0, 0.0, 0.0
    total_steps = 0

    for _ in range(sample_epochs):
        random.shuffle(samples)

        for i in range(batch_count):
            input_enc, input_dec = LearnInput(), LearnInput()
            mask, label_enc, label_dec = [], [], []

            start = batch_size * i
            for j in range(start, start + batch_size):
                sample = samples[j]
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
                batch_size, -1
            )
            label_tensor_enc = torch.tensor(
                label_enc, dtype=torch.float32, device=device
            ).view(batch_size, -1)
            label_tensor_dec = torch.tensor(
                label_dec, dtype=torch.float32, device=device
            ).view(batch_size, -1)

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
            loss_dec = loss_fn_dec(masked_logits, label_tensor_dec).mean()

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

            loss = loss_enc + loss_dec - (entropy_coef * entropy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Variance metric
            target_var = torch.var(label_tensor_enc)
            expl_var = (
                (1 - torch.var(label_tensor_enc - out_enc) / target_var).item()
                if target_var > 0
                else float("nan")
            )

            total_loss_enc += loss_enc.item()
            total_loss_dec += loss_dec.item()
            total_entropy += entropy.item()
            total_kl_div += kl_div.item()
            total_expl_var += expl_var
            total_steps += 1

    return {
        "loss_encoder": total_loss_enc / total_steps,
        "loss_decoder": total_loss_dec / total_steps,
        "entropy": total_entropy / total_steps,
        "kl_div": total_kl_div / total_steps,
        "expl_var": total_expl_var / total_steps,
    }


def main():
    results_dir = Path("results") / RUN_NAME
    results_dir.mkdir(exist_ok=True, parents=True)
    weights_dir = results_dir / Path("out")
    weights_dir.mkdir(exist_ok=True)
    vis_savedir = results_dir / Path("visuals")
    vis_savedir.mkdir(exist_ok=True)

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
    elo_data_path = results_dir / "elo.json"

    if model_path.exists():
        print("❇️ Restoring", model_path)
        model.load_state_dict(torch.load(model_path, weights_only=False))
    elif PRETRAIN_WEIGHTS_PATH.exists():
        print("❇️ Restoring", PRETRAIN_WEIGHTS_PATH)
        model.load_state_dict(torch.load(PRETRAIN_WEIGHTS_PATH, weights_only=False))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.CrossEntropyLoss(reduction="none")

    elo = EloRating(initial=600)
    if elo_data_path.exists():
        elo.load_json(elo_data_path)

    CARD_TYPE_WEIGHTS = {
        CardKind.STAGE2_POKEMON: 3.0,
        CardKind.STAGE1_POKEMON: 2.2,
        CardKind.BASIC_POKEMON: 1.2,
        CardKind.STADIUM: 1.7,
        CardKind.SPECIAL_ENERGY: 2.0,
        CardKind.SUPPORTER: 1.5,
        CardKind.TOOL: 1.2,
        CardKind.ITEM: 1.0,
        CardKind.BASIC_ENERGY: 1.0,
    }

    player1 = ISMCTSPlayer(
        model=model,
        name="ismcts",
        deck=mega_lucario_ex_deck,
        sampler=SoftWeightBayesianFilterSampler(
            deck_pool=[
                decks.alakazam_deck,
                decks.ionos_deck,
                decks.mega_lucario_ex_deck,
            ],
            deck_names=["alakazam", "ionos", "lucario"],
            card_role_weights={
                card_id: CARD_TYPE_WEIGHTS[kind]
                for card_id, kind in CARD_ID_TO_KIND.items()
            },
        ),
        sample_count=5,
        search_count_per_sample=70,
        batched_inference=True,
    )

    player_lucario = Player(
        model=player_lucario_agent,
        name="rule_based_lucario",
        deck=mega_lucario_ex_deck,
        is_trainable=False,
    )

    ctx = mp.get_context("spawn")
    model.share_memory()
    pool = ctx.Pool(processes=32)

    wandb.init(project="ptcg-rl", name=RUN_NAME, resume="allow")

    start_epoch = 0
    if (results_dir / "epoch.txt").exists():
        with open(results_dir / "epoch.txt", "r") as f:
            start_epoch = int(f.read())

    for _ in range(start_epoch):
        lr_scheduler.step()

    create_handle()
    ray.logger.setLevel("WARN")
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\n--- Epoch {epoch}: Data Collection ---", flush=True)
        samples, train_win_rate, sps = generate_samples(
            pool=pool,
            player1=player1,
            partner_player=player_lucario,
            num_games=TRAIN_ITERATIONS,
            vis_savedir=vis_savedir,
            epoch=epoch,
            elo=elo,
        )

        print(f"Train win rate: {train_win_rate}%", flush=True)
        elo.save_json(elo_data_path)

        if train_win_rate > SELF_PLAY_CLONE_UPDATE_WINRATE_THRESH:
            print("Updating checkpoint target model...", flush=True)
            model2.load_state_dict(model.state_dict())
            torch.save(model2.state_dict(), model2_path)

        print(
            f"--- Epoch {epoch}: Training on {len(samples)} samples across {SAMPLE_EPOCHS} pass(es) ---",
            flush=True,
        )
        metrics = train_on_samples(
            model=model,
            optimizer=optimizer,
            samples=samples,
            batch_size=BATCH_SIZE,
            sample_epochs=SAMPLE_EPOCHS,
            device=device,
            loss_fn_enc=loss_fn_enc,
            loss_fn_dec=loss_fn_dec,
            entropy_coef=ENTROPY_COEF,
        )

        if metrics:
            wandb.log(
                {
                    "train_win_rate": train_win_rate,
                    "sps": sps,
                    "elo": elo.ratings.get(player1.name, 600),
                    "loss_encoder": metrics["loss_encoder"],
                    "loss_decoder": metrics["loss_decoder"],
                    "loss_total": metrics["loss_encoder"] + metrics["loss_decoder"],
                    "entropy": metrics["entropy"],
                    "kl_div": metrics["kl_div"],
                    "expl_var": metrics["expl_var"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        torch.save(model.state_dict(), model_path)
        with open(results_dir / "epoch.txt", "w+") as f:
            f.write(str(epoch + 1))

        lr_scheduler.step()


if __name__ == "__main__":
    main()

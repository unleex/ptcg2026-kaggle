import torch
from ray import serve
import ray
from transformer_mcts import transformer


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    logging_config={"log_level": "WARN"},
    max_ongoing_requests=256,
)
class ModelServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = transformer.MyModel(
            d_model=256,
            num_heads=8,
            d_feedforward=1024,
            num_layers_encoder=4,
            num_layers_decoder=4,
        ).to(self.device)

    def batch_sparse_vectors(
        self,
        sv_list: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combines a list of SparseVectors into concatenated 1D tensors for EmbeddingBag."""
        flat_indices = []
        flat_values = []
        global_offsets = []

        current_element_count = 0
        for sv in sv_list:
            flat_indices.extend(sv.index)
            flat_values.extend(sv.value)
            # Shift sample offsets by the total number of elements processed so far
            for o in sv.offset:
                global_offsets.append(o + current_element_count)
            current_element_count += len(sv.index)

        return (
            torch.tensor(flat_indices, dtype=torch.int32).to(self.device),
            torch.tensor(flat_values, dtype=torch.float32).to(self.device),
            torch.tensor(global_offsets, dtype=torch.int32).to(self.device),
        )

    def pad_sv_dec(self, sv_dec, target_options=64):
        """Pads a decoder SparseVector to exactly 64 action options."""
        current_options = len(sv_dec.offset)
        if current_options >= target_options:
            return sv_dec

        indices = list(sv_dec.index)
        values = list(sv_dec.value)
        offsets = list(sv_dec.offset)

        # Pad missing action options with empty offsets
        for _ in range(target_options - current_options):
            offsets.append(len(indices))
        sv = transformer.SparseVector()
        sv.index = indices
        sv.value = values
        sv.offset = offsets
        return sv

    # Inside your ModelServer in inference_server.py:
    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.1)
    async def evaluate_states(self, state_tuples: list[tuple]):
        # 1. Pad decoder inputs to 64 options so every sample in the batch has identical shape
        enc_list = [s[0] for s in state_tuples]
        dec_list = [self.pad_sv_dec(s[1]) for s in state_tuples]

        # 2. Collate standard tensors
        enc_idx, enc_val, enc_off = self.batch_sparse_vectors(enc_list)
        dec_idx, dec_val, dec_off = self.batch_sparse_vectors(dec_list)

        # 3. Model forward pass (batch_size now evenly divides all bags)
        with torch.inference_mode():
            values, policies = self.model(
                enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off
            )

        # 4. Slice policy back to original action count for each worker
        values_cpu = values.squeeze(-1).tolist()
        policies_cpu = policies.tolist()

        results = []
        for i, (_, raw_sv_dec) in enumerate(state_tuples):
            num_actions = len(raw_sv_dec.offset)
            # Only return probabilities for actual valid actions
            valid_policy = policies_cpu[i][:num_actions]
            results.append((values_cpu[i], valid_policy))

        return results

    async def __call__(self, state_tuple: tuple) -> tuple[float, list[float]]:
        # Entry point for single-item requests from workers
        return await self.evaluate_states(state_tuple)


def create_handle():
    serve.run(ModelServer.bind(), name="model")

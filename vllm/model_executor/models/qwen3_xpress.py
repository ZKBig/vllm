# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.v1.worker.gpu.spec_decode.xpress import kernels

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .qwen3_dflash2 import CandidateSelector, DFlash2Qwen3DecoderLayer
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class XPressRefinerHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        block_size: int,
        rank: int = 256,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.rank = int(rank)
        r = self.rank
        self.w1 = nn.Embedding(vocab_size, r)
        self.down_h = nn.Linear(hidden_size, r, bias=False)
        self.down_g = nn.Linear(hidden_size, r, bias=False)
        self.in_proj = nn.Linear(3 * r, r, bias=False)
        # Stored FOLDED (L*tril + I), so a refine pass is one bmm with no mask and no
        # residual add. Identity here means "no mixing" for a head built without
        # weights.
        self.mix_L = nn.Parameter(torch.eye(block_size).expand(r, -1, -1).contiguous())
        self.mlp_gate = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, r, bias=False)
        self.w2 = nn.Linear(r, vocab_size, bias=False)
        self._scratch: dict = {}

    @torch.no_grad()
    def fold_from_raw_(self, raw_L: torch.Tensor) -> None:
        # Training stores the raw mixer and adds the sublayer residual: x + (L*tril)x.
        # Baking the mask and the identity into the parameter makes that one bmm, and
        # keeps a checkpoint meaning the same thing at serving time as it did in
        # training.
        B = self.block_size
        tril = torch.tril(torch.ones(B, B, dtype=raw_L.dtype, device=raw_L.device))
        eye = torch.eye(B, dtype=raw_L.dtype, device=raw_L.device)
        self.mix_L.copy_(raw_L * tril + eye)

    def hidden_cache(self, h_full: torch.Tensor) -> torch.Tensor:
        # Pass-invariant: only prev_ids changes between Jacobi passes, so compute the
        # hidden half once per block and reuse it for all K passes.
        g = h_full.mean(dim=1, keepdim=True).expand_as(h_full)
        return torch.cat([self.down_h(h_full), self.down_g(g)], dim=-1)

    def _refine_latent(
        self, prev_ids: torch.Tensor, hcache: torch.Tensor
    ) -> torch.Tensor:
        lat = self.w1(prev_ids)
        x = self.in_proj(torch.cat([hcache, lat], dim=-1))
        # Per-channel causal mix: position k sees only j <= k, which is what makes
        # Jacobi iteration valid -- a settled prefix cannot be disturbed by later slots.
        x = torch.bmm(self.mix_L.to(x.dtype), x.permute(2, 1, 0)).permute(2, 1, 0)
        return x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))

    def refine_bias(self, prev_ids: torch.Tensor, hcache: torch.Tensor) -> torch.Tensor:
        return self.w2(self._refine_latent(prev_ids, hcache))

    def jacobi_refine_greedy(
        self,
        base_logits_full: torch.Tensor,
        h_full: torch.Tensor,
        anchor_ids: torch.Tensor,
        tok_am1_ids: torch.Tensor,
        num_passes: int,
        seed_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Greedy, so a settled prefix stays settled and K passes converge monotonically.
        N, B, _ = base_logits_full.shape
        hcache = self.hidden_cache(h_full)
        blk = torch.empty(N, B, dtype=torch.long, device=h_full.device)
        blk[:, 0] = anchor_ids
        # seed_ids: the selector's greedy walk over the unary top-k, when the
        # checkpoint carries a candidate selector. It replaces the per-slot argmax
        # as the Jacobi starting point and is computed ONCE -- the candidate set is
        # never rescored between passes, so the refine loop stays fixed-shape.
        blk[:, 1:] = (
            seed_ids
            if seed_ids is not None
            else base_logits_full[:, 1:, :].argmax(dim=-1)
        )

        if not base_logits_full.is_cuda:
            # Reference path for the CPU unit tests, which run the head in float64
            # to pin the fold and causality invariants. Serving always takes the
            # fused path below.
            for _ in range(num_passes):
                prev = blk.roll(shifts=1, dims=1)
                prev[:, 0] = tok_am1_ids
                refined = base_logits_full + self.refine_bias(prev, hcache)
                blk[:, 1:] = refined[:, 1:, :].argmax(dim=-1)
            return blk[:, 1:]

        buf = self.fused_buffers()
        rows = N * (B - 1)
        v = base_logits_full.shape[-1]
        # ONE scratch set sized for the largest N seen. vLLM captures many batch
        # buckets, and a per-N cache would pin GBs that belong to the KV cache.
        if self._scratch.get("cap", 0) < N:
            nvb = (v + 4095) // 4096
            dev = base_logits_full.device
            dt = base_logits_full.dtype
            self._scratch = {
                "cap": N,
                "lat": torch.empty(N, B - 1, self.rank, dtype=dt, device=dev),
                "bias": torch.empty(rows, v, dtype=dt, device=dev),
                "base": torch.empty(rows, v, dtype=dt, device=dev),
                "ov": torch.empty(rows, nvb, dtype=torch.float32, device=dev),
                "oi": torch.empty(rows, nvb, dtype=torch.int64, device=dev),
            }
        sc = {
            k: (v_ if k == "cap" else v_[:N] if k == "lat" else v_[:rows])
            for k, v_ in self._scratch.items()
        }
        sc["base"].copy_(base_logits_full[:, 1:, :].reshape(rows, v))
        xh = torch.mm(hcache.view(N * B, -1), buf["whc_t"]).view(N, B, self.rank)
        # Three launches per pass: latent, the w2 GEMM, then add+argmax straight
        # into blk. The [N, B, V] sum is never materialized.
        for _ in range(num_passes):
            kernels.xpress_latent_pass(
                blk,
                tok_am1_ids,
                xh,
                sc["lat"],
                self.w1.weight,
                buf["wlat_t"],
                buf["mix_kjc"],
                buf["wg_t"],
                buf["wu_t"],
                buf["wd_t"],
            )
            torch.mm(sc["lat"].view(rows, self.rank), buf["w2_t"], out=sc["bias"])
            kernels.fused_add_argmax_to_blk(
                sc["base"], sc["bias"], sc["ov"], sc["oi"], blk
            )
        return blk[:, 1:]

    def fused_buffers(self) -> dict:
        if getattr(self, "_fused_buf", None) is None:
            r = self.rank
            w = self.in_proj.weight.detach()
            self._fused_buf = {
                "whc_t": w[:, : 2 * r].t().contiguous(),
                "wlat_t": w[:, 2 * r :].t().contiguous(),
                "mix_kjc": self.mix_L.detach().permute(1, 2, 0).contiguous(),
                "wg_t": self.mlp_gate.weight.detach().t().contiguous(),
                "wu_t": self.mlp_up.weight.detach().t().contiguous(),
                "wd_t": self.mlp_down.weight.detach().t().contiguous(),
                "w2_t": self.w2.weight.detach().t().contiguous(),
            }
        return self._fused_buf

    HYBRID_KEY_MAP = {
        "w1.weight": "w1.weight",
        "down_h.weight": "down_h.weight",
        "down_g.weight": "down_g.weight",
        "in_proj.weight": "in_proj.weight",
        "mix.L": "__raw_mix_L__",
        "mlp.gate_proj.weight": "mlp_gate.weight",
        "mlp.up_proj.weight": "mlp_up.weight",
        "mlp.down_proj.weight": "mlp_down.weight",
        "w2.weight": "w2.weight",
    }

    @torch.no_grad()
    def load_hybrid_state_dict(self, sd: dict) -> None:
        raw_L = None
        for src, dst in self.HYBRID_KEY_MAP.items():
            if src not in sd:
                raise KeyError(f"XPress head: missing key {src!r} in checkpoint")
            if dst == "__raw_mix_L__":
                raw_L = sd[src]
            else:
                p = dict(self.named_parameters())[dst]
                p.copy_(sd[src].to(p.dtype))
        if raw_L is None:
            raise KeyError("XPress head: checkpoint has no mixer weight to fold")
        self.fold_from_raw_(raw_L.to(self.mix_L.dtype))


class Qwen3XPressModel(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.xpress_head = XPressRefinerHead(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            block_size=getattr(config, "xpress_block_size", None)
            or (getattr(config, "num_speculative_steps", 15) + 1),
            rank=getattr(config, "xpress_rank", 256),
            mlp_hidden=getattr(config, "xpress_mlp_hidden", 512),
        )
        self.draft_vocab_size = draft_vocab_size
        if getattr(config, "xpress_compile_head", True):
            self.xpress_head.refine_bias = torch.compile(  # type: ignore[method-assign]
                self.xpress_head.refine_bias, dynamic=False
            )
            logger.info("XPress head refine_bias wrapped with torch.compile")


class Qwen3XPressForCausalLM(DFlashQwen3ForCausalLM):
    model_cls: type[Qwen3XPressModel] = Qwen3XPressModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise NotImplementedError(
                "XPress currently requires a full-vocab draft (the refiner bias "
                "is defined over the target vocabulary)."
            )
        self.draft_id_to_target_id = None

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def jacobi_refine_greedy(
        self,
        base_logits_full: torch.Tensor,
        h_full: torch.Tensor,
        anchor_ids: torch.Tensor,
        tok_am1_ids: torch.Tensor,
        num_passes: int,
    ) -> torch.Tensor:
        return self.model.xpress_head.jacobi_refine_greedy(
            base_logits_full, h_full, anchor_ids, tok_am1_ids, num_passes
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        raw_mix_L = None
        for name, loaded_weight in weights:
            if "t2d" in name or "d2t" in name:
                continue
            if name.startswith("xpress_head."):
                sub = name[len("xpress_head.") :]
                if sub == "mix.L":
                    raw_mix_L = loaded_weight
                    continue
                mapped = XPressRefinerHead.HYBRID_KEY_MAP.get(sub, sub)
                name = "model.xpress_head." + mapped
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # These are provided by the target (shared) or reconstructed below, so drop
        # them before the loader sees them rather than asking it to skip: the
        # mixer is stored raw in the checkpoint and folded after loading.
        skip_substrs = ["mask_embedding", "xpress_head.mix_L"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        model_weights = {
            k: v
            for k, v in model_weights.items()
            if not any(sub in k for sub in skip_substrs)
        }
        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights.items())
        if raw_mix_L is None:
            raise ValueError("XPress checkpoint is missing xpress_head.mix.L")
        self.model.xpress_head.fold_from_raw_(
            raw_mix_L.to(self.model.xpress_head.mix_L.dtype)
        )
        self.model._build_fused_kv_buffers()


class Qwen3XPressDFlash2Model(Qwen3XPressModel):
    """XPress refiner on the DFlash2 backbone: the same anchored-block decoder
    with a grouped dynamic convolution around each attention and MLP sublayer.
    Only the layer class changes; the refiner head and the Jacobi refinement are
    untouched.

    When the checkpoint was trained with ``xpress_selector``, DFlash2's bilinear
    candidate selector is loaded alongside the refiner and its greedy walk over
    the unary top-k replaces the per-slot argmax as the Jacobi seed. The walk runs
    ONCE per draft step; the candidate set is not rescored between passes, so the
    selector costs one pass worth of work no matter how large K is and the refine
    loop keeps the fixed shape the CUDA graph needs.
    """

    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        self.candidate_selector: CandidateSelector | None = None
        if getattr(self.config, "xpress_selector", False):
            # Its own tag, or it shares the draft head's compile cache.
            with set_model_tag("xpress_candidate_selector"):
                self.candidate_selector = CandidateSelector(
                    hidden_size=self.config.hidden_size,
                    vocab_size=self.config.vocab_size,
                    rank=int(self.config.selector_rank),
                    top_k=int(self.config.selector_top_k),
                    params_dtype=vllm_config.model_config.dtype,
                    prefix=maybe_prefix(prefix, "candidate_selector"),
                )

    def selector_seed(
        self,
        base_logits_full: torch.Tensor,  # [N, B, V]
        h_full: torch.Tensor,  # [N, B, H]
        anchor_ids: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """DFlash2's greedy selector walk over the unary top-k, at temperature 0.

        Mirrors the trainer's ``XPressModel._selector_seed`` exactly: per slot,
        score this slot's top-k against the token the walk chose for the previous
        slot and keep the best. Slot 0 of the block is the anchor, which is given,
        so the walk starts at slot 1. B-1 iterations of small fixed-shape ops --
        unrolled into the draft step's graph, like the Jacobi passes themselves.
        """
        selector = self.candidate_selector
        assert selector is not None
        logits = base_logits_full[:, 1:, :]  # drop the anchor slot
        k = selector.top_k
        candidates = logits.topk(k, dim=-1).indices  # [N, B-1, k]
        unary = logits.gather(-1, candidates).float()  # [N, B-1, k]
        proj = selector.hidden_projection(
            h_full[:, 1:, :].to(selector.hidden_projection.weight.dtype)
        )  # [N, B-1, r]
        out = torch.empty_like(candidates[..., 0])
        prev = anchor_ids
        for slot in range(out.shape[1]):
            context = selector.predecessor_codebook[prev] * proj[:, slot]
            successors = selector.successor_codebook[candidates[:, slot]]
            trans = (context.unsqueeze(-2) * successors).sum(dim=-1)
            best = (unary[:, slot] + trans.float()).argmax(dim=-1)
            prev = candidates[:, slot].gather(1, best.unsqueeze(1)).squeeze(1)
            out[:, slot] = prev
        return out


class Qwen3XPressDFlash2ForCausalLM(Qwen3XPressForCausalLM):
    model_cls = Qwen3XPressDFlash2Model

    def jacobi_refine_greedy(
        self,
        base_logits_full: torch.Tensor,
        h_full: torch.Tensor,
        anchor_ids: torch.Tensor,
        tok_am1_ids: torch.Tensor,
        num_passes: int,
    ) -> torch.Tensor:
        seed_ids = None
        if self.model.candidate_selector is not None:
            seed_ids = self.model.selector_seed(base_logits_full, h_full, anchor_ids)
        return self.model.xpress_head.jacobi_refine_greedy(
            base_logits_full,
            h_full,
            anchor_ids,
            tok_am1_ids,
            num_passes,
            seed_ids=seed_ids,
        )

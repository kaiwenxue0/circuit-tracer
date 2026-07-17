"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import contextlib
import logging
import time
import weakref
from functools import partial
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import torch.nn.functional as F
import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.disk_offload import offload_modules


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            activation_matrix, error_vectors, token_vectors, decoder_vecs, feature_output_hook
        )

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

    def _caching_hooks(self, feature_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> Tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            proxy._batch_buffer[write_index] = einsum(
                grads.to(output_vecs.dtype)[read_index],
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    def _make_attribution_hooks(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """Create the complete backward-hook for computing attribution scores."""

        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[start:end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return activation_matrix._nnz() + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        # Token-embedding nodes
        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[tok_start : tok_start + n_pos],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),
            bwd_hooks=self._attribution_hooks,
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[int(layer)].register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]

class AttributionContextForLlada:
    """
    Same role as AttributionContext, but modified for diffusion LLMs (bi-directional attention).
    Key differences:
       • backward is applied to the *scalar target logit*
       • gradient hooks only read the target_pos location
       • prevents global averaging that kills feature influence
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> None:

        n_layers, n_pos, _ = activation_matrix.shape

        self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        total_active_feats = activation_matrix._nnz()
        self._row_size = total_active_feats + (n_layers + 1) * n_pos

        self._activation_matrix = activation_matrix
        self._decoder_vecs = decoder_vecs
        self._error_vectors = error_vectors
        self._token_vectors = token_vectors
        self._feature_output_hook = feature_output_hook

    # ---------- Forward caching hooks ----------
    def _caching_hooks(self, feature_input_hook: str) -> List[Tuple[str, Callable]]:

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int):
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    # ---------- Backward score hook factory (modified for LLADA) ----------
    def _compute_score_hook_targetpos(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        target_pos: int,
    ) -> Tuple[str, Callable]:

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            # grads: (batch, pos, d_model)
            # Only read gradient from target_pos (not all positions)
            g = grads.to(output_vecs.dtype)[:, target_pos]  # (batch, d_model)
            # Contract: (batch, d_model) x (position, d_model) -> (position, batch)
            proxy._batch_buffer[write_index] = einsum(
                g,
                output_vecs,
                "batch d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    # ---------- Installing hooks ----------
    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel", target_pos: int):
        """Install forward cache + backward attribution hooks."""
        n_layers, n_pos, _ = self._activation_matrix.shape
        nnz_layers, nnz_positions, _ = self._activation_matrix.indices()

        total_feats = self._activation_matrix._nnz()

        # map each layer → [start:end] in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # ------ construct ALL attribution hooks ------
        hooks = []

        # Feature nodes
        for layer, (start, end) in enumerate(layer_spans):
            if start == end:
                continue
            hooks.append(
                self._compute_score_hook_targetpos(
                    f"blocks.{layer}.{self._feature_output_hook}",
                    self._decoder_vecs[start:end],    # (num_feat_layer, d_model)
                    np.s_[start:end],                 # write these rows
                    target_pos,
                )
            )

        # Error nodes
        def err_offset(l): return total_feats + l * n_pos
        for layer in range(self.n_layers):
            hooks.append(
                self._compute_score_hook_targetpos(
                    f"blocks.{layer}.{self._feature_output_hook}",
                    self._error_vectors[layer],       # (n_pos, d_model)
                    np.s_[err_offset(layer):err_offset(layer + 1)],
                    target_pos,
                )
            )

        # Token nodes
        tok_start = total_feats + self.n_layers * n_pos
        hooks.append(
            self._compute_score_hook_targetpos(
                "hook_embed",
                self._token_vectors,                 # (n_pos, d_model)
                np.s_[tok_start:tok_start + n_pos],
                target_pos,
            )
        )

        # Store
        self._attribution_hooks = hooks

        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),
            bwd_hooks=self._attribution_hooks,
        ):
            yield

    # ---------- Compute one batch of rows ----------
    def compute_batch(
        self,
        model,
        logits: torch.Tensor,
        target_pos: int,
        target_token: int,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:

        batch = len(layers)
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch,
            device=inject_values.device,
            dtype=inject_values.dtype,
        )

        # Inject custom gradients into the saved residual activations
        batch_idx = torch.arange(batch, device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = (layers == layer)
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(
                self._resid_activations[int(layer)].register_hook(fn)
            )

        # ---------- *** Key change: backward from the scalar target logit *** ----------
        res = self._resid_activations[-1]
        assert res is not None, "Final residual not cached (forward hooks missing)."
        # Direction = embedding vector of target token  W_U[:, target_token]
        direction = model.unembed.W_U[:, target_token].to(res.dtype)   # (d_model,)
        direction = direction.view(1, -1)
        # Build gradient that only applies at target_pos
        grad = torch.zeros_like(res)            # (batch, seq, d_model)
        grad[batch_idx, target_pos] = direction        # inject only target position
        try:
            res.backward(gradient=grad, retain_graph=retain_graph)
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[:batch]


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


@torch.no_grad()
def select_scaled_decoder_vecs(
    activations: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    """

    device = activations.device
    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activations):
        _, feat_idx = row.coalesce().indices()
        W_dec = transcoders[layer].W_dec.to(device)
        rows.append(W_dec[feat_idx])
    values = activations.values().to(device)[:, None]
    return torch.cat(rows) * values


@torch.no_grad()
def select_encoder_rows(
    activation_matrix: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""

    device = activation_matrix.device
    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        _, feat_idx = row.coalesce().indices()
        W_enc = transcoders[layer].W_enc.T
        W_enc = W_enc.to(device) 
        rows.append(W_enc[feat_idx])
    return torch.cat(rows)


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def ensure_tokenized(prompt: Union[str, torch.Tensor, List[int]], tokenizer) -> torch.Tensor:
    """Convert *prompt* → 1-D tensor of token ids (no batch dim)."""

    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt").input_ids[0]
    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0) if prompt.ndim == 2 else prompt
    if isinstance(prompt, list):
        return torch.tensor(prompt, dtype=torch.long)
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    model_type="MDM"
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        if model_type == "ARM":
            return _run_attribution(
                model=model,
                prompt=prompt,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                verbose=verbose,
                offload_handles=offload_handles,
                update_interval=update_interval,
                logger=logger,
            )
        elif model_type == "MDM":
            return _run_attribution_llada(
                model=model,
                prompt=prompt,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                verbose=verbose,
                offload_handles=offload_handles,
                update_interval=update_interval,
                logger=logger,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        logger.removeHandler(handler)


def _run_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    update_interval=4,
    logger=None,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = ensure_tokenized(prompt, model.tokenizer)
    logits, activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        input_ids, sparse=True
    )
    # print("activation_matrix:", activation_matrix)
    decoder_vecs = select_scaled_decoder_vecs(activation_matrix, model.transcoders)
    encoder_rows = select_encoder_rows(activation_matrix, model.transcoders)
    ctx = AttributionContext(
        activation_matrix, error_vecs, token_vecs, decoder_vecs, model.feature_output_hook
    )
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {activation_matrix._nnz()} active features")
    print(f"Found {activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    logger.info(
        f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=encoder_rows[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

# if gen_length == steps, model generate one token per step.
def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def pick_first_generated_positions(
    model,
    x,                         # shape: (1, L_prompt+gen_length) with mask_id 填好
    prompt_len: int,           # L_prompt
    block_length: int,
    steps_per_block: int,     
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
):
    # x = prompt + gen_length * [mask]
    device = x.device
    assert x.shape[0] == 1

    prompt_index = (x != mask_id)

    block_mask_index = (x[:, prompt_len: prompt_len + block_length:] == mask_id)  # (1, block_length)
    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
    mask_index = (x == mask_id)
    if cfg_scale > 0.:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_ = torch.cat([x, un_x], dim=0)
        logits = model(x_)
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(x)
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0_p[:, prompt_len + block_length:] = -np.inf
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, 0])
        transfer_index[j, select_index] = True
    
    return transfer_index

def _run_attribution_llada(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    update_interval=4,
    logger=None,
    steps=8,
    gen_length=8,
    block_length=1,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Attribution for LLaDA with mask-based generation:
    - 只对“第一次生成的 token”（首次被填的 mask 位置）做 attribution
    """

    start_time = time.time()

    # ---------- Phase 0: 预处理 ----------
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()

    input_ids_1d = ensure_tokenized(prompt, model.tokenizer)
    device = next(model.parameters()).device
    def ensure_batch(x, device):
        if isinstance(x, list):
            x = torch.tensor(x, device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x.to(device)
    
    input_ids_2d = ensure_batch(input_ids_1d, device)     # (1, L_prompt)

    # prompt + [MASK]*gen_length
    x = torch.full((1, input_ids_2d.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :input_ids_2d.shape[1]] = input_ids_2d.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # setup_attribution_llada 期望 1D ids
    ids_1d = x.squeeze(0)  # (L_prompt + gen_length,)

    logits, activation_matrix, error_vecs, token_vecs = model.setup_attribution_llada(
        ids_1d, sparse=True
    )
    decoder_vecs = select_scaled_decoder_vecs(activation_matrix, model.transcoders)
    encoder_rows = select_encoder_rows(activation_matrix, model.transcoders)

    ctx = AttributionContextForLlada(
        activation_matrix, error_vecs, token_vecs, decoder_vecs, model.feature_output_hook
    )
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {activation_matrix._nnz()} active features")
    
    print(f"Found {activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # ------------------ Phase 1: determine target_pos EARLY ------------------ 
    logger.info("Phase 1: Determining first generated token position")
    phase_start = time.time()
    first_targets = pick_first_generated_positions(
        model=model,
        x=x,                               # (1, L)
        prompt_len=input_ids_2d.shape[1],
        block_length=block_length,
        steps_per_block=steps_per_block,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
    )
    print("first_targets:", first_targets)
    _, target_pos = torch.where(first_targets)
    target_pos = target_pos.item()
    logger.info(f"First generated token position = {target_pos}")

    # ------------------ Phase 2: Forward pass with hooks ------------------
    logger.info("Phase 2: Forward pass with hooks")
    phase_start = time.time()
    with ctx.install_hooks(model, target_pos=target_pos):
        residual = model.forward(ids_1d.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    
    # ------------------ Phase 3: Build node groups, salient logits ------------------
    logger.info("Phase 3: Build input vectors / salient logits")
    phase_start = time.time()

    n_layers, n_pos, _ = activation_matrix.shape
    feat_layers, feat_pos, _ = activation_matrix.indices()
    total_active_feats = activation_matrix._nnz()

    # print("n_layers:", n_layers)
    # print("n_pos:", n_pos)
    # print("n_pos_tokens:", ids_1d.shape[0])

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        logits[0, target_pos],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    logger.info(f"Top logits selected: {len(logit_idx)}, prob sum={logit_p.sum().item():.4f}")
    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    n_features = total_active_feats
    n_errors   = n_layers * n_pos
    n_tokens   = ids_1d.shape[0]
    logit_offset = n_features + n_errors + n_tokens
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits
    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)

    logger.info(f"Node segmentation: features={n_features}, errors={n_errors}, tokens={n_tokens}, logits={n_logits}")
    logger.info(f"Total nodes={total_nodes}, edge_matrix={edge_matrix.shape}")
    logger.info(f"Phase 3 time={time.time()-phase_start:.2f}s")

    # ------------------ Phase 4: logit attribution ------------------
    logger.info("Phase 4: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, n_logits, batch_size):
        batch = logit_vecs[i : i + batch_size]
        # rows = ctx.compute_batch(
        #     layers=torch.full((batch.shape[0],), n_layers),
        #     positions=torch.full((batch.shape[0],), target_pos),
        #     inject_values=batch,
        # )
        target_tokens = logit_idx[i : i + batch_size]
        rows = ctx.compute_batch(
            model=model,
            logits=logits,
            target_pos=target_pos,
            target_token=target_tokens,   # 每个 logit 对应不同 token
            layers=torch.full((batch.shape[0],), n_layers, device=batch.device),
            positions=torch.full((batch.shape[0],), target_pos, device=batch.device),
            inject_values=batch,
        )

        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # ------------------ Phase 5: feature attribution ------------------
    logger.info("Phase 5: Feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            # rows = ctx.compute_batch(
            #     layers=feat_layers[idx_batch],
            #     positions=feat_pos[idx_batch],
            #     inject_values=encoder_rows[idx_batch],
            #     retain_graph=n_visited < max_feature_nodes,
            # )
            rows = ctx.compute_batch(
                model=model,
                logits=logits,
                target_pos=target_pos,
                target_token=logit_idx[0],     # 只对 top1 logit 做归因（你之前也是）
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=encoder_rows[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # ------------------ Phase 6: Package graph ------------------
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(ids_1d),
        input_tokens=ids_1d,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
        target_ctx_idx=target_pos,
    )

    total_time = time.time() - start_time
    logger.info(f"LLaDA attribution completed in {total_time:.2f}s")
    return graph

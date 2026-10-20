"""
Example for exporting LiteTracker sub-components to ONNX format and using with a wrapper.
The three neural components use ONNX Runtime; preprocessing and state still use
CPU PyTorch. ManagedLiteTracker supplies stable IDs and keyframe session handovers.

Usage:
    # Export only:
    uv run onnx_example.py --checkpoint path/to/weights.pth --output_dir weights/onnx_managed

    # Export + verify against PyTorch:
    uv run onnx_example.py --checkpoint path/to/weights.pth --output_dir weights/onnx_managed --verify
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.lite_tracker import LiteTracker, posenc
from src.managed_lite_tracker import ManagedLiteTracker
from src.model_utils import (
    bilinear_sampler,
    get_1d_sincos_pos_embed_from_grid,
    sample_features5d,
)


def load_model(checkpoint_path: str) -> LiteTracker:
    """
    Load a LiteTracker model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth).

    Returns:
        LiteTracker: The loaded model in evaluation mode.
    """
    model = LiteTracker()
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_fnet(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export the feature encoder (BasicEncoder) to ONNX.

    Creates a dummy input matching the model's expected resolution and exports
    the feature extraction CNN with dynamic batch size support.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX file.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        str: Path to the exported ONNX file.
    """
    output_path = os.path.join(output_dir, "fnet.onnx")
    H, W = model.model_resolution
    dummy_input = torch.randn(1, 3, H, W)

    torch.onnx.export(
        model.fnet,
        dummy_input,
        output_path,
        opset_version=opset_version,
        dynamo=False,
        input_names=["frame"],
        output_names=["features"],
        dynamic_axes={
            "frame": {0: "batch"},
            "features": {0: "batch"},
        },
    )
    print(f"  Exported fnet -> {output_path}")
    return output_path


def export_corr_mlp(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export the correlation MLP to ONNX.

    The MLP takes a flattened correlation volume (r^2 * r^2 values, where r = 2*corr_radius+1)
    and produces a compressed correlation embedding.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX file.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        str: Path to the exported ONNX file.
    """
    output_path = os.path.join(output_dir, "corr_mlp.onnx")
    r = 2 * model.corr_radius + 1
    input_size = r * r * r * r  # 49 * 49 = 2401
    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        model.corr_mlp,
        dummy_input,
        output_path,
        opset_version=opset_version,
        dynamo=False,
        input_names=["corr_volume"],
        output_names=["corr_emb"],
        dynamic_axes={
            "corr_volume": {0: "num_points"},
            "corr_emb": {0: "num_points"},
        },
    )
    print(f"  Exported corr_mlp -> {output_path}")
    return output_path


def export_updateformer(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export the EfficientUpdateFormer transformer to ONNX.

    The transformer takes per-point, per-frame tokens and an attention mask, and
    outputs coordinate/visibility/confidence deltas.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX file.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        str: Path to the exported ONNX file.
    """
    output_path = os.path.join(output_dir, "updateformer.onnx")

    B, N, T = 1, 100, 4
    dummy_input = torch.randn(B, N, T, model.input_dim)
    dummy_mask = torch.ones(B, T, N, dtype=torch.bool)

    torch.onnx.export(
        model.updateformer,
        (dummy_input, dummy_mask),
        output_path,
        opset_version=opset_version,
        dynamo=False,
        input_names=["input_tensor", "mask"],
        output_names=["delta"],
        dynamic_axes={
            "input_tensor": {0: "batch", 1: "num_points", 2: "window_size"},
            "mask": {0: "batch", 1: "window_size", 2: "num_points"},
            "delta": {0: "batch", 1: "num_points", 2: "window_size"},
        },
    )
    print(f"  Exported updateformer -> {output_path}")
    return output_path


def export_all(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export all LiteTracker sub-components to ONNX.

    Exports fnet, corr_mlp, and updateformer as separate ONNX files into the
    specified output directory.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX files.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        dict: Mapping from component name to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Exporting LiteTracker sub-components to ONNX...")
    paths = {
        "fnet": export_fnet(model, output_dir, opset_version),
        "corr_mlp": export_corr_mlp(model, output_dir, opset_version),
        "updateformer": export_updateformer(model, output_dir, opset_version),
    }
    config = {
        name: getattr(model, name)
        for name in (
            "window_len",
            "stride",
            "corr_radius",
            "corr_levels",
            "model_resolution",
            "iters",
            "input_dim",
            "latent_dim",
            "inv_sigmoid_true_val",
        )
    }
    config["format_version"] = 1
    Path(output_dir, "tracker_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    np.save(Path(output_dir, "time_emb.npy"), model.time_emb.detach().cpu().numpy())
    print("Export complete!")
    return paths


# ---------------------------------------------------------------------------
# ONNX-based LiteTracker wrapper
# ---------------------------------------------------------------------------


class OnnxLiteTracker:
    """
    A drop-in replacement for LiteTracker that uses ONNX Runtime for inference.

    This class replicates the full stateful tracking pipeline of LiteTracker.forward(),
    but uses ONNX models for the three neural network forward passes (fnet, corr_mlp,
    updateformer). All stateful logic (buffers, flow estimation, etc.) is handled in
    numpy/torch.

    Usage:
        tracker = OnnxLiteTracker("weights/onnx_managed")
        for frame in video_frames:
            coords, vis, conf = tracker(frame_tensor, queries_tensor)
    """

    def __init__(
        self,
        onnx_dir: str,
        window_len: int | None = None,
        stride: int | None = None,
        corr_radius: int | None = None,
        corr_levels: int | None = None,
        model_resolution: tuple | None = None,
        iters: int | None = None,
        input_dim: int | None = None,
        latent_dim: int | None = None,
        providers: list | None = None,
        *,
        threads: int = 2,
    ):
        """Load the original three-graph export or a new export with metadata.

        Original positional/keyword arguments remain supported. Omitted settings
        come from export metadata when present, otherwise from the original
        defaults. Explicit settings must agree with metadata to avoid silent
        mismatches. Neural components use ONNX Runtime; host state and sampling
        still use CPU PyTorch tensors. Mac defaults to CPU, CUDA when available.
        """
        import onnxruntime as ort

        defaults = {
            "window_len": 16,
            "stride": 4,
            "corr_radius": 3,
            "corr_levels": 4,
            "model_resolution": (384, 512),
            "iters": 1,
            "input_dim": 1110,
            "latent_dim": 128,
        }
        requested = {
            "window_len": window_len,
            "stride": stride,
            "corr_radius": corr_radius,
            "corr_levels": corr_levels,
            "model_resolution": model_resolution,
            "iters": iters,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
        }
        config_path = Path(onnx_dir, "tracker_config.json")
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        if config and config.get("format_version") != 1:
            raise ValueError("Unsupported tracker export format; re-export the model.")
        for name, default in defaults.items():
            exported = config.get(name, default)
            value = requested[name]
            if name == "model_resolution":
                exported = tuple(exported)
                value = tuple(value) if value is not None else None
            if config and value is not None and value != exported:
                raise ValueError(f"{name} does not match the exported configuration.")
            setattr(self, name, exported if value is None else value)
        self.inv_sigmoid_true_val = config.get("inv_sigmoid_true_val", 4.6)
        if providers is None:
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()
            ]
        if threads < 1:
            raise ValueError("threads must be positive.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        for name in ("fnet", "corr_mlp", "updateformer"):
            setattr(
                self,
                name + "_session",
                ort.InferenceSession(
                    str(Path(onnx_dir, name + ".onnx")),
                    sess_options=options,
                    providers=providers,
                ),
            )
        if config:
            self.time_emb = torch.from_numpy(
                np.load(Path(onnx_dir, "time_emb.npy"), allow_pickle=False)
            )
        else:
            # Legacy export folders contain only the three ONNX files.
            time_grid = torch.linspace(0, self.window_len - 1, self.window_len).reshape(
                1, self.window_len, 1
            )
            self.time_emb = get_1d_sincos_pos_embed_from_grid(
                self.input_dim, time_grid[0]
            )
        if self.time_emb.shape != (1, self.window_len, self.input_dim):
            raise ValueError(
                "Time embedding does not match the exported configuration."
            )
        self.reset()

    def reset(self):
        """
        Reset all internal state, buffers, and caches for a new video sequence.

        Call this method before processing a new video to clear all temporal
        state from the previous sequence.
        """
        self.online_ind = 0
        self.ema_flow_buffer = None
        self.corr_embs_buffer = None
        self.coords_buffer = None
        self.vis_buffer = None
        self.conf_buffer = None
        self.track_feat_cache = [None] * self.corr_levels

    def _run_fnet(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Run the feature encoder via ONNX Runtime.

        Args:
            frame (torch.Tensor): Normalized input frame of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Feature map of shape [B, latent_dim, H//stride, W//stride].
        """
        result = self.fnet_session.run(None, {"frame": frame.numpy()})
        return torch.from_numpy(result[0])

    def _run_corr_mlp(self, corr_volume: torch.Tensor) -> torch.Tensor:
        """
        Run the correlation MLP via ONNX Runtime.

        Args:
            corr_volume (torch.Tensor): Flattened correlation volume of shape [N_points, r^4].

        Returns:
            torch.Tensor: Correlation embeddings of shape [N_points, 256].
        """
        result = self.corr_mlp_session.run(None, {"corr_volume": corr_volume.numpy()})
        return torch.from_numpy(result[0])

    def _run_updateformer(
        self, input_tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the update transformer via ONNX Runtime.

        Args:
            input_tensor (torch.Tensor): Transformer input of shape [B, N, T, input_dim].
            mask (torch.Tensor): Attention mask of shape [B, T, N] (bool).

        Returns:
            torch.Tensor: Delta predictions of shape [B, N, T, 4] (dx, dy, dvis, dconf).
        """
        result = self.updateformer_session.run(
            None,
            {
                "input_tensor": input_tensor.numpy(),
                "mask": mask.numpy(),
            },
        )
        return torch.from_numpy(result[0])

    def _interpolate_time_embed(self, t: int) -> torch.Tensor:
        """
        Interpolate the temporal positional embedding to match the current window size.

        Uses linear interpolation when the current window size differs from the
        precomputed embedding length.

        Args:
            t (int): Target temporal length.

        Returns:
            torch.Tensor: Interpolated time embedding of shape [1, t, input_dim].
        """
        T = self.time_emb.shape[1]
        if t == T:
            return self.time_emb
        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb

    def _get_support_points(self, coords, r, reshape_back=True):
        """
        Generate a grid of support points around each coordinate for local feature sampling.

        Mirrors LiteTracker.get_support_points() exactly.

        Args:
            coords (torch.Tensor): Input coordinates of shape [B, T, N, 3].
            r (int): Radius for the support grid.
            reshape_back (bool): Whether to reshape the output for downstream use.

        Returns:
            torch.Tensor: Support points for each coordinate.
        """
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)
        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)
        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack((zgrid, xgrid, ygrid), dim=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl
        if reshape_back:
            third_dim = int((2 * r + 1) ** 2)
            return coords_lvl.reshape(B, N, third_dim, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def _get_track_feat(self, fmaps, queried_coords, support_radius=0):
        """
        Extract track features at the queried coordinates and their support points.

        Mirrors LiteTracker.get_track_feat() exactly.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, C, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.
            support_radius (int): Radius for support points.

        Returns:
            tuple: (central track features, all support track features).
        """
        sample_coords = torch.cat(
            [
                torch.zeros_like(queried_coords[..., :1][:, None]),
                queried_coords[:, None],
            ],
            dim=-1,
        )
        support_points = self._get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def _get_correlation_feat(self, fmaps, queried_coords):
        """
        Compute correlation features for the queried coordinates using bilinear sampling.

        Mirrors LiteTracker.get_correlation_feat() exactly.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, D, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.

        Returns:
            torch.Tensor: Correlation features of shape [B, T, N, r, r, D].
        """
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self._get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor, queries: torch.Tensor):
        """
        Predict tracks for the given frame and queries.

        This replicates LiteTracker.forward() but uses ONNX models for neural network
        forward passes. All stateful orchestration (buffer management, flow momentum,
        coordinate initialization) is handled identically to the PyTorch version.

        Args:
            frame (torch.Tensor): Input frame of shape [B, C, H, W] (float32, 0-255 range).
            queries (torch.Tensor): Point queries of shape [B, N, 3]; first channel is the
                frame index, the rest are (x, y) coordinates.

        Returns:
            tuple:
                coords (torch.Tensor): Predicted coordinates [B, 1, N, 2].
                vis (torch.Tensor): Predicted visibility mask [B, 1, N].
                conf (torch.Tensor): Predicted confidence [B, 1, N].
        """
        if frame.device.type != "cpu" or queries.device.type != "cpu":
            raise ValueError("This hybrid ONNX backend requires CPU input tensors.")
        if frame.dtype != torch.float32 or queries.dtype != torch.float32:
            raise ValueError("This ONNX export requires float32 frames and queries.")
        original_shape = frame.shape
        frame = F.interpolate(
            frame, size=self.model_resolution, mode="bilinear", align_corners=True
        )
        queries_scaled = queries.clone()
        queries_scaled[:, :, 1] *= (self.model_resolution[1] - 1) / (
            original_shape[3] - 1
        )  # W
        queries_scaled[:, :, 2] *= (self.model_resolution[0] - 1) / (
            original_shape[2] - 1
        )  # H

        B, _C, H, W = frame.shape
        B, N, __ = queries_scaled.shape
        frame = 2 * (frame / 255.0) - 1.0

        T = 1

        queried_frames = queries_scaled[:, :, 0].long()

        # Downscale the query coords for the smaller size feat maps
        queried_coords = queries_scaled[..., 1:3].clone()
        queried_coords = queried_coords / self.stride

        # Run feature encoder via ONNX
        fmaps = self._run_fnet(frame.float())
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )

        # Build feature pyramid
        fmaps_pyramid = [fmaps]
        track_feat_support_pyramid = []
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        is_track_initialized_now = (queried_frames == self.online_ind)[
            :, None, :, None
        ]  # B 1 N 1

        for i in range(self.corr_levels):
            if self.online_ind == 0:
                _, track_feat_support = self._get_track_feat(
                    fmaps_pyramid[i],
                    queried_coords / 2**i,
                    support_radius=self.corr_radius,
                )
                self.track_feat_cache[i] = torch.zeros_like(track_feat_support)
                self.track_feat_cache[i] += (
                    track_feat_support * is_track_initialized_now
                )
            else:
                if is_track_initialized_now.any():
                    _, track_feat_support = self._get_track_feat(
                        fmaps_pyramid[i],
                        queried_coords / 2**i,
                        support_radius=self.corr_radius,
                    )
                    self.track_feat_cache[i] += (
                        track_feat_support * is_track_initialized_now
                    )

            track_feat_support_pyramid.append(self.track_feat_cache[i].unsqueeze(1))

        # Initialize vis and conf for the current frame with zeros
        vis_init = torch.zeros((B, T, N, 1)).float()
        conf_init = torch.zeros((B, T, N, 1)).float()
        coords_init = queried_coords.reshape(B, T, N, 2).float()

        vis_init = torch.where(
            is_track_initialized_now.expand_as(vis_init),
            self.inv_sigmoid_true_val,
            vis_init,
        )
        conf_init = torch.where(
            is_track_initialized_now.expand_as(conf_init),
            self.inv_sigmoid_true_val,
            conf_init,
        )

        if self.online_ind == 0:
            self.ema_flow_buffer = torch.zeros_like(coords_init)

        # Handle tracks that are initialized in the previous frame.
        is_track_previsouly_initialized = (queried_frames < self.online_ind)[
            :, None, :, None
        ]  # B 1 N 1
        if self.online_ind > 0:
            vis_init = torch.where(
                is_track_previsouly_initialized.expand_as(vis_init),
                self.vis_buffer[:, -1],
                vis_init,
            )
            conf_init = torch.where(
                is_track_previsouly_initialized.expand_as(conf_init),
                self.conf_buffer[:, -1],
                conf_init,
            )
            # If there is only one frame processed so far, we initialize the coordinates
            # with the previous frame's coordinates
            if self.online_ind == 1:
                coords_init = torch.where(
                    is_track_previsouly_initialized.expand_as(coords_init),
                    self.coords_buffer[:, -1],
                    coords_init,
                )
            # If there is more, we use the exponential moving average of the flow
            else:
                last_flow = self.coords_buffer[:, -1] - self.coords_buffer[:, -2]
                cached_flow = self.ema_flow_buffer
                alpha = 0.8
                accumulated_flow = alpha * last_flow + (1 - alpha) * cached_flow
                self.ema_flow_buffer = accumulated_flow
                coords_init = torch.where(
                    is_track_previsouly_initialized.expand_as(coords_init),
                    self.coords_buffer[:, -1] + accumulated_flow,
                    coords_init,
                )

        # forward_window logic
        coords, viss, confs = self._forward_window(
            fmaps_pyramid=fmaps_pyramid,
            coords=coords_init,
            track_feat_support_pyramid=track_feat_support_pyramid,
            queried_frames=queried_frames,
            vis=vis_init,
            conf=conf_init,
            iters=self.iters,
            is_track_previsouly_initialized=is_track_previsouly_initialized,
        )

        coords[:, :, :, 0] *= (original_shape[3] - 1) / (
            self.model_resolution[1] - 1
        )  # W
        coords[:, :, :, 1] *= (original_shape[2] - 1) / (
            self.model_resolution[0] - 1
        )  # H

        viss = torch.sigmoid(viss)
        confs = torch.sigmoid(confs)

        viss = viss * confs
        thr = 0.6
        viss = viss > thr

        self.online_ind += 1
        return coords, viss, confs

    def _forward_window(
        self,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        queried_frames,
        vis,
        conf,
        is_track_previsouly_initialized,
        iters=4,
    ):
        """
        Run the tracking update for a window of frames using ONNX sub-models.

        Replicates LiteTracker.forward_window() exactly, substituting PyTorch
        model calls with ONNX Runtime inference for corr_mlp and updateformer.

        Args:
            fmaps_pyramid (list[torch.Tensor]): Feature maps at different scales [B, T, C, H, W].
            coords (torch.Tensor): Track coordinates for each frame [B, T, N, 2].
            track_feat_support_pyramid (list[torch.Tensor]): Template features at different
                scales [B, 1, r^2, N, C].
            queried_frames (torch.Tensor): Frame indices of the queries [B, N].
            vis (torch.Tensor): Visibility logits for tracks [B, T, N, 1].
            conf (torch.Tensor): Confidence logits for tracks [B, T, N, 1].
            is_track_previsouly_initialized (torch.Tensor): Mask for tracks initialized in
                previous frames [B, 1, N, 1].
            iters (int): Number of update iterations.

        Returns:
            tuple: (coords, vis, conf) for the current window.
        """
        device = fmaps_pyramid[0].device
        B = fmaps_pyramid[0].shape[0]
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        num_new_frames = 1
        num_prev_frames = min(self.online_ind, (self.window_len - 1))
        current_window_size = (
            num_prev_frames + num_new_frames
        )  # total number of frames in the current window

        # Compute the frame indices for the current window
        left_ind = max(0, self.online_ind - self.window_len + 1)  # inclusive
        right_ind = self.online_ind + 1  # not inclusive
        frame_indices = (
            torch.arange(left_ind, right_ind, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, -1, N)
        )  # shape [B, T, N]

        # `attention_mask` is a boolean mask: True if the track is initialized at or before
        # this frame
        attention_mask = (
            queried_frames.unsqueeze(1).expand(B, -1, N) <= frame_indices
        )  # B T N

        corr_embs = torch.empty(1, device=device)
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(-1, N, 2)
            # Extract correlation embeddings from the new frames
            corr_embs_list = []
            for i in range(self.corr_levels):
                corr_feat = self._get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                # Run corr_mlp via ONNX
                corr_flat = corr_volume.reshape(
                    B * num_new_frames * N, r * r * r * r
                ).float()
                corr_embs_list.append(self._run_corr_mlp(corr_flat))
            corr_embs = torch.cat(corr_embs_list, dim=-1).view(B, num_new_frames, N, -1)

            # If it is the first time step, we skip the transformer as there is nothing
            # to compute, simply use the computed `corr_embs` as well as the initial
            # values of `coords`, `vis` and `conf` to initialize the buffers.
            if self.online_ind == 0:
                break

            prev_coords = self.coords_buffer.detach()
            prev_vis = self.vis_buffer.detach()
            prev_conf = self.conf_buffer.detach()
            prev_corr_embs = self.corr_embs_buffer.detach()

            current_window_coords = torch.cat([prev_coords, coords], dim=1)
            current_window_vis = torch.cat([prev_vis, vis], dim=1)
            current_window_conf = torch.cat([prev_conf, conf], dim=1)
            current_window_corr_embs = torch.cat([prev_corr_embs, corr_embs], dim=1)

            transformer_input = [
                current_window_vis,
                current_window_conf,
                current_window_corr_embs,
            ]
            rel_coords_forward = (
                current_window_coords[:, :-1] - current_window_coords[:, 1:]
            )
            rel_coords_backward = (
                current_window_coords[:, 1:] - current_window_coords[:, :-1]
            )
            rel_coords_forward = F.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = F.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]], device=device
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale
            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)
            x = (
                (torch.cat(transformer_input, dim=-1))
                .permute(0, 2, 1, 3)
                .reshape(B * N, current_window_size, -1)
            )
            x = x + self._interpolate_time_embed(current_window_size)
            x = x.view(B, N, current_window_size, -1)

            # Run updateformer via ONNX
            delta = self._run_updateformer(x.float(), attention_mask)
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            # Update the values of the current frame only for the points that are
            # initialized before this frame.
            vis[is_track_previsouly_initialized] = (
                vis[is_track_previsouly_initialized]
                + delta_vis[:, -num_new_frames:][is_track_previsouly_initialized]
            )
            conf[is_track_previsouly_initialized] = (
                conf[is_track_previsouly_initialized]
                + delta_conf[:, -num_new_frames:][is_track_previsouly_initialized]
            )
            coords[is_track_previsouly_initialized.expand_as(coords)] = (
                coords[is_track_previsouly_initialized.expand_as(coords)]
                + delta_coords[:, -num_new_frames:][
                    is_track_previsouly_initialized.expand_as(coords)
                ]
            )

        # Update buffers
        if self.online_ind == 0:
            self.coords_buffer = coords
            self.vis_buffer = vis
            self.conf_buffer = conf
            self.corr_embs_buffer = corr_embs
        else:
            self.coords_buffer = torch.cat([self.coords_buffer, coords], dim=1)
            self.vis_buffer = torch.cat([self.vis_buffer, vis], dim=1)
            self.conf_buffer = torch.cat([self.conf_buffer, conf], dim=1)
            self.corr_embs_buffer = torch.cat([self.corr_embs_buffer, corr_embs], dim=1)

        if current_window_size == self.window_len:
            self.coords_buffer = self.coords_buffer[:, 1:]
            self.vis_buffer = self.vis_buffer[:, 1:]
            self.conf_buffer = self.conf_buffer[:, 1:]
            self.corr_embs_buffer = self.corr_embs_buffer[:, 1:]

        coords = coords[..., :2] * float(self.stride)
        vis = vis[..., 0]
        conf = conf[..., 0]
        return coords, vis, conf


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_export(checkpoint_path: str, onnx_dir: str, num_frames: int = 36):
    """Exercise ManagedLiteTracker with both backends, failing on any mismatch.

    Uses the same externally chosen queries/events in both runs to isolate backend
    equivalence. Checks dynamic query counts, future births, rolling history,
    keyframe priming, empty sessions, and a fresh video reset. This is numerical
    equivalence testing, not an accuracy benchmark or independent policy comparison.
    """
    torch.set_num_threads(2)
    pt_model = load_model(checkpoint_path)
    onnx_model = OnnxLiteTracker(onnx_dir, providers=["CPUExecutionProvider"])
    for name in (
        "window_len",
        "stride",
        "corr_radius",
        "corr_levels",
        "model_resolution",
        "iters",
    ):
        if getattr(pt_model, name) != getattr(onnx_model, name):
            raise ValueError(f"Export configuration mismatch: {name}")
    boundary = pt_model.window_len + 4
    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    full_lifecycle = num_frames >= boundary + 3
    if not full_lifecycle:
        print(
            f"Short verification: use at least {boundary + 3} frames to include a keyframe handover."
        )
    pt = ManagedLiteTracker(pt_model, max_points=24)
    ort = ManagedLiteTracker(onnx_model, max_points=24)
    generator = torch.Generator().manual_seed(42)
    base = torch.randint(0, 256, (1, 3, 480, 640), generator=generator).float()
    points = torch.tensor(
        [[[x, y] for y in (150.0, 300.0) for x in (180.0, 250.0, 320.0, 390.0, 460.0)]]
    )
    queries = torch.cat((torch.zeros_like(points[..., :1]), points), dim=2)
    report = {
        "passed": True,
        "full_lifecycle_coverage": full_lifecycle,
        "provider": "CPUExecutionProvider",
        "coordinate_atol_pixels": 0.01,
        "confidence_atol": 0.001,
        "max_coordinate_error": 0.0,
        "max_confidence_error": 0.0,
        "visibility_mismatches": 0,
        "checks": [],
        "events": [],
    }

    def compare(label, actual, expected):
        finite = all(torch.isfinite(x).all().item() for x in (*actual, *expected))
        coordinate_error = (
            (actual[0] - expected[0]).abs().max().item() if actual[0].numel() else 0.0
        )
        confidence_error = (
            (actual[2] - expected[2]).abs().max().item() if actual[2].numel() else 0.0
        )
        mismatches = (actual[1] != expected[1]).sum().item()
        state_matches = (
            pt.track_ids == ort.track_ids
            and pt.frame_index == ort.frame_index
            and pt.session_index == ort.session_index
            and pt_model.online_ind == onnx_model.online_ind
        )
        history_matches = (
            all(
                getattr(pt_model, name).shape == getattr(onnx_model, name).shape
                for name in pt._CACHES
            )
            if pt_model.online_ind
            else True
        )
        ok = (
            finite
            and coordinate_error <= report["coordinate_atol_pixels"]
            and confidence_error <= report["confidence_atol"]
            and mismatches == 0
            and state_matches
            and history_matches
        )
        report["passed"] &= ok
        report["max_coordinate_error"] = max(
            report["max_coordinate_error"], coordinate_error
        )
        report["max_confidence_error"] = max(
            report["max_confidence_error"], confidence_error
        )
        report["visibility_mismatches"] += mismatches
        report["checks"].append(
            {
                "label": label,
                "passed": ok,
                "points": len(pt.track_ids),
                "coordinate_error": coordinate_error,
                "confidence_error": confidence_error,
                "visibility_mismatches": mismatches,
            }
        )
        print(
            f"{label}: {'PASS' if ok else 'FAIL'} | {len(pt.track_ids)} points | "
            f"xy error={coordinate_error:.6f}px | confidence error={confidence_error:.6f} | "
            f"visibility mismatches={mismatches}",
            flush=True,
        )

    for i in range(num_frames):
        frame = base.roll(shifts=(i // 2, i), dims=(-2, -1))
        if i == 3:
            new_queries = torch.tensor([[[5.0, 220.0, 200.0], [5.0, 420.0, 240.0]]])
            pt_ids = pt.add_queries(new_queries)
            ort_ids = ort.add_queries(new_queries)
            if pt_ids != ort_ids:
                raise AssertionError("Backend query IDs differ after insertion.")
            report["events"].append(
                {"frame": i, "kind": "append_future_queries", "birth_frame": 5}
            )
        actual = pt(frame, queries if i == 0 else None)
        expected = ort(frame, queries if i == 0 else None)
        compare(f"frame_{i}", actual, expected)
        if i == boundary and full_lifecycle:
            carry = []
            for j, tid in enumerate(pt.track_ids):
                valid = all(
                    out[1][0, 0, j].item()
                    and torch.isfinite(out[0][0, 0, j]).all().item()
                    and (out[0][0, 0, j] >= 0).all().item()
                    and out[0][0, 0, j, 0] < 640
                    and out[0][0, 0, j, 1] < 480
                    for out in (actual, expected)
                )
                if valid:
                    carry.append(tid)
            carry = carry[:4]
            if not carry:
                raise RuntimeError(
                    "No common visible points to exercise keyframe ID carryover."
                )
            new_points = torch.tensor(
                [[[240.0, 180.0], [350.0, 260.0], [440.0, 320.0]]]
            )
            compare(
                "keyframe_prime",
                pt.start_keyframe(carry, new_points),
                ort.start_keyframe(carry, new_points),
            )
            report["events"].append(
                {"frame": i, "kind": "keyframe", "carried_ids": carry}
            )
    compare("empty_boundary", pt.start_keyframe([]), ort.start_keyframe([]))
    compare("empty_frame", pt(base), ort(base))
    new_queries = torch.tensor([[[float(pt.frame_index), 320.0, 240.0]]])
    pt_ids = pt.add_queries(new_queries)
    ort_ids = ort.add_queries(new_queries)
    if pt_ids != ort_ids:
        raise AssertionError("Backend query IDs differ after empty session.")
    compare("after_empty", pt(base), ort(base))
    pt.reset()
    ort.reset()
    compare("new_video", pt(base, queries[:, :3]), ort(base, queries[:, :3]))
    path = Path("results") / Path(onnx_dir).resolve().name / "verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verification report: {path.resolve()}")
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--checkpoint", required=True)
    parser.add_argument("-o", "--output_dir", default="weights/onnx_managed")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify managed query lifecycle against PyTorch.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing export without exporting again.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=36,
        help="Verification frames; at least 23 for a handover with the default model.",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not args.verify_only:
        model = load_model(args.checkpoint)
        export_all(model, args.output_dir, opset_version=args.opset)
        del model
    if (args.verify or args.verify_only) and not verify_export(
        args.checkpoint, args.output_dir, args.num_frames
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

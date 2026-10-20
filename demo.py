import os
from contextlib import nullcontext

# LiteTracker uses 3D grid sampling, which PyTorch 2.7.1 cannot run on MPS.
# Keep supported operations on MPS and allow this operation to use CPU.
# This must be set before importing torch; preserve explicit user settings.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import argparse
import imageio.v3 as iio
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

from src.model_utils import get_points_on_a_grid
from src.execution_timer import ExecutionTimer, LogExecutionTime
from src.lite_tracker import LiteTracker
from src.visualizer import Visualizer


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video_path",
        required=True,
        help="path to a video",
    )
    parser.add_argument(
        "-w",
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "-s", "--grid_size", type=int, default=10, help="Regular grid size"
    )
    parser.add_argument(
        "-q",
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    # Arrange the model and queries
    model = LiteTracker()
    with open(args.checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    window_frames = []

    def _process_step(frame):
        with torch.no_grad():
            frame = (
                torch.tensor(frame, device=device)
                .permute(2, 0, 1)[None]
                .float()
            )  # shape is (B, C, H, W)

            return model(
                frame,
                queries=queries,
            )

    pred_tracks = []
    pred_visibility = []

    # Get the total number of frames from the video:
    from imageio_ffmpeg import count_frames_and_secs

    num_frames, _ = count_frames_and_secs(args.video_path)
    print(f"Total frames in the video: {num_frames}")

    # Use mixed precision only on CUDA GPUs that support BF16.
    # MPS and CPU run in float32 without entering a CUDA context.
    with (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device == "cuda" and dtype == torch.bfloat16
        else nullcontext()
    ):
        # Iterating over video frames, processing one window at a time:
        for i, frame in tqdm(
            enumerate(
                iio.imiter(
                    args.video_path,
                    plugin="FFMPEG",
                )
            )
        ):
            if i == 0:
                assert args.grid_size > 0, "Grid size should be positive"
                H = frame.shape[0]
                W = frame.shape[1]
                grid_pts = get_points_on_a_grid(args.grid_size, (H, W))
                queries = torch.cat(
                    [
                        torch.ones_like(grid_pts[:, :, :1]) * args.grid_query_frame,
                        grid_pts,
                    ],
                    dim=2,
                ).to(device)

            with LogExecutionTime("Track frame"):
                coords, viss, confs = _process_step(
                    frame,
                )
            pred_tracks.append(coords)
            pred_visibility.append(viss)
            window_frames.append(frame)

    ExecutionTimer.print_stats()
    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=device).permute(
        0, 3, 1, 2
    )[None]
    # Get the video fps
    in_video = cv2.VideoCapture(args.video_path)
    fps = int(in_video.get(cv2.CAP_PROP_FPS))
    in_video.release()
    vis = Visualizer(save_dir="./results", pad_value=120, linewidth=3, fps=fps)
    filename = Path(args.video_path).stem
    pred_tracks = torch.cat(pred_tracks, dim=1)
    pred_visibility = torch.cat(pred_visibility, dim=1)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=args.grid_query_frame,
        filename=filename,
    )

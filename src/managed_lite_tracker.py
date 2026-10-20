"""Point lifecycle operations around LiteTracker; no mapping or pose policy."""

from typing import Protocol

import torch


class TrackerBackend(Protocol):
    """State contract; backend caches hold tensors after the first inference."""

    online_ind: int
    stride: int
    model_resolution: tuple[int, int]
    coords_buffer: torch.Tensor | None
    vis_buffer: torch.Tensor | None
    conf_buffer: torch.Tensor | None
    corr_embs_buffer: torch.Tensor | None
    ema_flow_buffer: torch.Tensor | None
    track_feat_cache: list[torch.Tensor | None]

    def reset(self) -> None: ...

    def __call__(
        self, frame: torch.Tensor, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class ManagedLiteTracker(torch.nn.Module):
    """Single-stream inference with stable IDs and bounded live point state.

    Supply initial queries to ``forward(frame, queries)`` or ``add_queries``.
    Subsequent calls normally use ``forward(frame)``. Add queries between calls;
    query times use the absolute input frame index, starting at zero. Outputs
    retain LiteTracker's tuple/shapes and follow the current ``track_ids``.

    Lifecycle decisions are Python-side, outside a future exported neural graph.
    Cache operations below support tensor-backed PyTorch and hybrid ONNX backends.
    Queries cannot be deleted within a session, including while invisible.
    start_keyframe() replaces the whole session at the last processed image.
    Set device/eval mode before starting; use reset() explicitly for a new video.
    """

    _CACHES = (
        "coords_buffer",
        "vis_buffer",
        "conf_buffer",
        "corr_embs_buffer",
        "ema_flow_buffer",
    )

    def __init__(self, model: TrackerBackend, max_points: int = 512):
        super().__init__()
        if model.online_ind != 0:
            raise ValueError("Use a fresh or reset LiteTracker instance.")
        self.model = model
        if max_points < 1:
            raise ValueError("max_points must be positive.")
        self.max_points = max_points
        self._clear_state()

    def _clear_state(self):
        self._track_ids: list[int] = []
        self._next_track_id = 0
        self._queries: torch.Tensor | None = None
        self._frame_index = 0
        self._backend_frame_offset = 0
        self._session_index = 0
        self._last_keyframe_frame = None
        self._last_frame = None
        self._last_coords = None
        self._last_visible = None
        self._last_prediction_ids = []
        self._image_shape = None

    @property
    def frame_index(self) -> int:
        """Absolute index of the next input frame, including empty frames."""
        return self._frame_index

    @property
    def session_index(self) -> int:
        """Tracker session number; independent of the absolute input clock."""
        return self._session_index

    @property
    def track_ids(self) -> list[int]:
        """Snapshot of IDs in output order; carried IDs survive session boundaries."""
        return self._track_ids.copy()

    @property
    def queries(self):
        return None if self._queries is None else self._queries.clone()

    def reset(self):
        """Explicitly begin a new independent video (IDs restart at zero)."""
        self.model.reset()
        self._clear_state()

    @torch.no_grad()
    def start_keyframe(
        self, keep_ids: list[int], new_points: torch.Tensor | None = None
    ):
        """End the session and prime a fresh one on the last processed image.

        keep_ids select visible, finite, in-bounds observations from that image.
        Their latest coordinates and external IDs are carried; neural histories
        are not. new_points is [1, M, 2] in input-image pixels and gets new IDs.
        Returns the usual prediction tuple for the SAME absolute frame. The next
        forward() must receive the following video frame, not this image again.
        This is a whole-session boundary, never an in-session cache deletion.
        """
        if self._last_frame is None:
            raise ValueError("Process a frame before starting a keyframe session.")
        boundary = self.frame_index - 1
        if self._last_keyframe_frame == boundary:
            raise ValueError("This frame already started a keyframe session.")
        if self.track_ids != self._last_prediction_ids:
            raise ValueError(
                "Process pending query additions before a session boundary."
            )
        if len(keep_ids) != len(set(keep_ids)) or any(
            tid not in self._last_prediction_ids for tid in keep_ids
        ):
            raise ValueError("Carry IDs must be unique IDs from the last prediction.")
        indices = [self._last_prediction_ids.index(tid) for tid in keep_ids]
        points = self._last_coords[:, 0, indices].clone()
        if not self._last_visible[0, 0, indices].all():
            raise ValueError("Only currently visible observations can be carried.")
        if new_points is None:
            new_points = points.new_empty((1, 0, 2))
        if new_points.ndim != 3 or new_points.shape[0] != 1 or new_points.shape[2] != 2:
            raise ValueError("Expected new_points shape [1, M, 2].")
        if new_points.device != points.device or new_points.dtype != points.dtype:
            raise ValueError("New points must match prediction device and dtype.")
        count = new_points.shape[1]
        points = torch.cat((points, new_points), dim=1)
        if points.shape[1] > self.max_points:
            raise ValueError("Keyframe queries would exceed max_points.")
        queries = torch.cat((torch.full_like(points[..., :1], boundary), points), dim=2)
        self._validate_queries(queries)
        self._check_bounds(queries, self._image_shape)
        # Validate everything before discarding the previous session's caches.
        frame = self._last_frame
        new_ids = list(range(self._next_track_id, self._next_track_id + count))
        self.model.reset()
        self._session_index += 1
        self._last_keyframe_frame = boundary
        self._backend_frame_offset = boundary
        self._frame_index = boundary
        self._track_ids = list(keep_ids) + new_ids
        self._next_track_id += count
        self._queries = queries.detach().clone()
        return self.forward(frame)

    def _remember_prediction(self, frame, outputs):
        # One image is retained so a boundary uses exactly the image in which
        # these coordinates were measured, even if the caller reuses its buffer.
        self._last_frame = frame.detach().clone()
        self._last_coords = outputs[0].detach().clone()
        self._last_visible = outputs[1].detach().clone()
        self._last_prediction_ids = self.track_ids
        return outputs

    @staticmethod
    def _validate_queries(queries):
        if queries.ndim != 3 or queries.shape[0] != 1 or queries.shape[2] != 3:
            raise ValueError("Expected queries shape [1, N, 3].")
        if not queries.is_floating_point() or not torch.isfinite(queries).all():
            raise ValueError("Queries must be finite floating-point values.")
        times = queries[..., 0]
        if not torch.equal(times, times.floor()):
            raise ValueError("Query start frames must be integers.")

    @torch.no_grad()
    def add_queries(self, queries: torch.Tensor) -> list[int]:
        """Append queries starting on the next input frame or later.

        Returns new IDs. At capacity, raises rather than silently dropping points.
        Coordinates are pixels in the input image, which must keep a fixed size.
        """
        self._validate_queries(queries)
        if (queries[..., 0] < self.frame_index).any():
            raise ValueError("New queries cannot start on an already processed frame.")
        count = queries.shape[1]
        if len(self._track_ids) + count > self.max_points:
            raise ValueError("Adding queries would exceed max_points.")
        if self._queries is not None and (
            queries.device != self._queries.device
            or queries.dtype != self._queries.dtype
        ):
            raise ValueError("New queries must match the existing device and dtype.")
        if self._image_shape is not None:
            self._check_bounds(queries, self._image_shape)
        if self.model.online_ind > 0 and count:
            self._append_cache_points(queries)
        new_ids = list(range(self._next_track_id, self._next_track_id + count))
        self._next_track_id += count
        self._track_ids.extend(new_ids)
        self._queries = (
            queries.detach().clone()
            if self._queries is None
            else torch.cat((self._queries, queries.detach()), dim=1)
        )
        return new_ids

    @staticmethod
    def _check_bounds(queries, image_shape):
        h, w = image_shape
        xy = queries[..., 1:]
        if (xy < 0).any() or (xy[..., 0] >= w).any() or (xy[..., 1] >= h).any():
            raise ValueError("Query coordinates must be inside the input image.")

    @torch.no_grad()
    def forward(self, frame: torch.Tensor, queries: torch.Tensor | None = None):
        """Return coords [1,1,N,2], visibility [1,1,N], confidence [1,1,N].

        Optional queries initialize the stream; thereafter, if supplied, they
        must equal the current managed queries. Use add_queries for insertion.
        Capture track_ids alongside each output before subsequent mutations.
        """
        if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[1] != 3:
            raise ValueError("Expected one RGB frame [1, 3, H, W].")
        shape = tuple(frame.shape[-2:])
        if min(shape) < 2:
            raise ValueError("Image dimensions must be at least two pixels.")
        if self._image_shape is not None and self._image_shape != shape:
            raise ValueError("Input image size must stay fixed during a stream.")
        if queries is not None:
            self._validate_queries(queries)
            if queries.device != frame.device:
                raise ValueError("Frame and queries must be on the same device.")
            self._check_bounds(queries, shape)
            if self._queries is None:
                self.add_queries(queries)
            elif not torch.equal(queries, self._queries):
                raise ValueError(
                    "Pass current managed queries, or omit queries; use add_queries to insert."
                )
        if self._queries is not None and self._queries.device != frame.device:
            raise ValueError("Frame and queries must be on the same device.")
        if self._queries is not None and self._image_shape is None:
            self._check_bounds(self._queries, shape)
        self._image_shape = shape
        if not self._track_ids:
            # No meaningful point history exists during this gap. Keep the backend
            # clock paused and translate future query times; never reset the model.
            self._frame_index += 1
            self._backend_frame_offset += 1
            return self._remember_prediction(
                frame,
                (
                    frame.new_empty((1, 1, 0, 2), dtype=torch.float32),
                    torch.empty((1, 1, 0), dtype=torch.bool, device=frame.device),
                    frame.new_empty((1, 1, 0), dtype=torch.float32),
                ),
            )

        backend_queries = self._queries.clone()
        backend_queries[..., 0] -= self._backend_frame_offset
        coords, visible, confidence = self.model(frame, queries=backend_queries)
        active = self._queries[..., 0] <= self.frame_index
        # Future queries are not observations, even if the backend reports them.
        visible = visible & active[:, None, :]
        self._frame_index += 1
        return self._remember_prediction(frame, (coords, visible, confidence))

    def _append_cache_points(self, queries):
        """Extend point axes without changing existing values or temporal length.

        Old coordinates for a new point are filled with its query location so its
        first flow estimate is zero. Pre-birth history is masked by LiteTracker.
        Appearance templates stay zero until the query's actual start frame.
        """
        count = queries.shape[1]
        h, w = self._image_shape
        xy = queries[..., 1:].clone()
        xy[..., 0] *= (self.model.model_resolution[1] - 1) / (w - 1) / self.model.stride
        xy[..., 1] *= (self.model.model_resolution[0] - 1) / (h - 1) / self.model.stride
        for name in self._CACHES:
            old = getattr(self.model, name)
            padding = old.new_zeros((old.shape[0], old.shape[1], count, old.shape[3]))
            if name == "coords_buffer":
                padding[:] = xy[:, None].to(old.dtype)
            setattr(self.model, name, torch.cat((old, padding), dim=2))
        self.model.track_feat_cache = [
            torch.cat(
                (f, f.new_zeros((f.shape[0], f.shape[1], count, f.shape[3]))), dim=2
            )
            for f in self.model.track_feat_cache
        ]

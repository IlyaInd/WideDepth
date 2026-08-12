"""NumPy-first loader for the WideDepth indoor benchmark."""

from __future__ import annotations

import numbers
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, get_args

import cv2
import numpy as np

BenchmarkTask = Literal[
    "mono",
    "depth_completion",
    "stereo_vertical",
    "stereo_horizontal",
]
ImageType = Literal["fisheye", "pano", "pano_crop", "pinhole", "pinhole_crop"]


@dataclass(frozen=True)
class _SamplePaths:
    scene_id: str
    image: Path
    target: Path
    right_image: Optional[Path] = None


class WideDepthBenchmarkDataset:
    """Access one configuration of the WideDepth indoor benchmark.

    The loader preserves the stored image geometry: it does not crop, resize,
    or rotate samples. RGB arrays are returned in RGB channel order as
    ``uint8``. Depth is returned in metres and disparity in pixels, both as
    ``float32``. A value of zero is treated as invalid.

    Args:
        root_dir: Extracted WideDepth benchmark root containing scene folders.
        task: One of ``mono``, ``depth_completion``, ``stereo_vertical``, or
            ``stereo_horizontal``.
        image_type: Released projection/view representation to load.
        fov: Camera field of view in degrees.
        baseline_mm: Stereo baseline in millimetres. It is part of the release
            directory layout for every task.
        sparse_ratio: Fraction of valid dense-depth pixels retained for the
            simulated sparse input for the ``depth_completion`` task.
        seed: Non-negative seed used for deterministic depth sparsification.

    Returns:
        ``mono`` samples contain ``sample_id``, ``image``, ``depth``, and
        ``valid_mask``. ``depth_completion`` adds ``sparse_depth`` while
        retaining dense ``depth`` as supervision. Stereo samples contain
        ``sample_id``, ``left_image``, ``right_image``, ``disparity``, and
        ``valid_mask``.
    """

    SUPPORTED_TASKS = frozenset(get_args(BenchmarkTask))
    SUPPORTED_FOVS = frozenset({120, 140, 165, 195})
    SUPPORTED_BASELINES_MM = frozenset({20, 65, 120, 200, 300})

    _IMAGE_TYPES_BY_FOV = {
        120: frozenset({"fisheye", "pano", "pano_crop", "pinhole", "pinhole_crop"}),
        140: frozenset({"fisheye", "pano", "pano_crop"}),
        165: frozenset({"fisheye", "pano", "pano_crop"}),
        195: frozenset({"fisheye", "pano", "pano_crop"}),
    }
    _STEREO_TASKS = frozenset({"stereo_vertical", "stereo_horizontal"})
    _RIGHT_CAMERA_BY_TASK = {
        "stereo_vertical": "DOWN",
        "stereo_horizontal": "RIGHT",
    }
    _SCENE_NAME = re.compile(r"^\d{3}_\d{3}_\d{3}$")
    _DEPTH_SCALE = 1000.0
    _DISPARITY_SCALE = 100.0

    def __init__(
        self,
        root_dir: Union[str, Path],
        task: BenchmarkTask,
        image_type: ImageType = "pano_crop",
        fov: int = 120,
        baseline_mm: int = 65,
        sparse_ratio: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.task = task
        self.image_type = image_type
        self.fov = fov
        self.baseline_mm = baseline_mm
        self.sparse_ratio = self._validate_sparse_ratio(sparse_ratio)
        self.seed = self._validate_seed(seed)

        self._validate_root()
        self._validate_request()
        self.samples, self.skipped_scenes = self._discover_samples()
        self.scene_ids: Tuple[str, ...] = tuple(
            sample.scene_id for sample in self.samples
        )

        if not self.samples:
            raise FileNotFoundError(
                "No complete WideDepth benchmark samples found for "
                f"task={task!r}, image_type={image_type!r}, fov={fov}, "
                f"baseline_mm={baseline_mm} under {self.root_dir}"
            )

        # The public release is missing some requested view files in scenes 096
        # and 097. Keep valid configurations usable and make every omission
        # visible instead of failing later during iteration.
        if self.skipped_scenes:
            preview = ", ".join(self.skipped_scenes[:5])
            suffix = "" if len(self.skipped_scenes) <= 5 else ", ..."
            warnings.warn(
                f"The benchmark root lacks required files for {len(self.skipped_scenes)} "
                f"scene(s) in the requested configuration; omitted: {preview}{suffix}",
                RuntimeWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        index = self._normalize_index(index)
        paths = self.samples[index]
        image = self._read_rgb(paths.image)

        if self.task in self._STEREO_TASKS:
            if paths.right_image is None:
                raise RuntimeError(
                    f"Internal error: missing stereo path for {paths.scene_id}"
                )
            right_image = self._read_rgb(paths.right_image)
            disparity = self._read_scaled_map(
                paths.target,
                scale=self._DISPARITY_SCALE,
                target_name="disparity",
            )
            self._validate_alignment(paths.scene_id, image, disparity, right_image)
            return {
                "sample_id": paths.scene_id,
                "left_image": image,
                "right_image": right_image,
                "disparity": disparity,
                "valid_mask": disparity > 0,
            }

        depth = self._read_scaled_map(
            paths.target,
            scale=self._DEPTH_SCALE,
            target_name="depth",
        )
        self._validate_alignment(paths.scene_id, image, depth)
        valid_mask = depth > 0

        if self.task == "mono":
            return {
                "sample_id": paths.scene_id,
                "image": image,
                "depth": depth,
                "valid_mask": valid_mask,
            }

        sparse_depth = self._sparsify_depth(depth, valid_mask, index)
        return {
            "sample_id": paths.scene_id,
            "image": image,
            "sparse_depth": sparse_depth,
            "depth": depth,
            "valid_mask": valid_mask,
        }

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"WideDepth benchmark root does not exist: {self.root_dir}"
            )
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"WideDepth benchmark root is not a directory: {self.root_dir}"
            )

    def _validate_request(self) -> None:
        if not isinstance(self.task, str) or self.task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task {self.task!r}; expected one of {sorted(self.SUPPORTED_TASKS)}"
            )
        if self.fov not in self.SUPPORTED_FOVS:
            raise ValueError(
                f"Unsupported fov {self.fov!r}; expected one of {sorted(self.SUPPORTED_FOVS)}"
            )
        if self.baseline_mm not in self.SUPPORTED_BASELINES_MM:
            raise ValueError(
                "Unsupported baseline_mm "
                f"{self.baseline_mm!r}; expected one of {sorted(self.SUPPORTED_BASELINES_MM)}"
            )

        allowed_image_types = self._IMAGE_TYPES_BY_FOV[self.fov]
        if (
            not isinstance(self.image_type, str)
            or self.image_type not in allowed_image_types
        ):
            raise ValueError(
                f"Unsupported image_type {self.image_type!r} for fov={self.fov}; "
                f"expected one of {sorted(allowed_image_types)}"
            )
        if self.task in self._STEREO_TASKS and self.image_type == "fisheye":
            raise ValueError(
                "image_type='fisheye' is unavailable for stereo tasks because the "
                "benchmark release does not provide fisheye disparity"
            )

    def _discover_samples(self) -> Tuple[Tuple[_SamplePaths, ...], Tuple[str, ...]]:
        scene_dirs = tuple(
            path
            for path in sorted(
                self.root_dir.iterdir(), key=lambda candidate: candidate.name
            )
            if path.is_dir() and self._SCENE_NAME.fullmatch(path.name)
        )
        if not scene_dirs:
            raise FileNotFoundError(
                f"No published WideDepth scene directories found under {self.root_dir}"
            )

        samples = []
        skipped = []
        for scene_dir in scene_dirs:
            sample = self._paths_for_scene(scene_dir)
            required_paths = [sample.image, sample.target]
            if sample.right_image is not None:
                required_paths.append(sample.right_image)
            if all(path.is_file() for path in required_paths):
                samples.append(sample)
            else:
                skipped.append(scene_dir.name)

        return tuple(samples), tuple(skipped)

    def _paths_for_scene(self, scene_dir: Path) -> _SamplePaths:
        configuration_dir = scene_dir / f"{self.fov}FOV" / f"{self.baseline_mm}mm"
        center_dir = configuration_dir / "CENTER"
        image_path = center_dir / f"{self.image_type}.png"

        if self.task in self._STEREO_TASKS:
            right_camera = self._RIGHT_CAMERA_BY_TASK[self.task]
            return _SamplePaths(
                scene_id=scene_dir.name,
                image=image_path,
                right_image=configuration_dir / right_camera / f"{self.image_type}.png",
                target=center_dir / "disparity" / f"{self.image_type}.png",
            )

        return _SamplePaths(
            scene_id=scene_dir.name,
            image=image_path,
            target=center_dir / "depth" / f"{self.image_type}.png",
        )

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Failed to decode RGB image: {path}")
        if (
            image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError(
                f"Expected uint8 three-channel RGB image at {path}, got "
                f"dtype={image_bgr.dtype}, shape={image_bgr.shape}"
            )
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_scaled_map(path: Path, scale: float, target_name: str) -> np.ndarray:
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None:
            raise OSError(f"Failed to decode {target_name} map: {path}")
        if encoded.ndim != 2 or not np.issubdtype(encoded.dtype, np.integer):
            raise ValueError(
                f"Expected a single-channel integer {target_name} map at {path}, got "
                f"dtype={encoded.dtype}, shape={encoded.shape}"
            )
        decoded = encoded.astype(np.float32) / np.float32(scale)
        if not np.isfinite(decoded).all():
            raise ValueError(
                f"Decoded {target_name} contains non-finite values: {path}"
            )
        return decoded

    @staticmethod
    def _validate_alignment(
        scene_id: str,
        image: np.ndarray,
        target: np.ndarray,
        right_image: Optional[np.ndarray] = None,
    ) -> None:
        expected_shape = image.shape[:2]
        if target.shape != expected_shape:
            raise ValueError(
                f"Spatial mismatch in scene {scene_id}: image shape={expected_shape}, "
                f"target shape={target.shape}"
            )
        if right_image is not None and right_image.shape[:2] != expected_shape:
            raise ValueError(
                f"Spatial mismatch in scene {scene_id}: left shape={expected_shape}, "
                f"right shape={right_image.shape[:2]}"
            )

    def _sparsify_depth(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        index: int,
    ) -> np.ndarray:
        rng = np.random.default_rng([self.seed, index])
        selected = (
            rng.random(depth.shape, dtype=np.float32) < self.sparse_ratio
        ) & valid_mask
        sparse_depth = np.zeros_like(depth)
        sparse_depth[selected] = depth[selected]
        return sparse_depth

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, numbers.Integral):
            raise TypeError(
                f"Dataset index must be an integer, got {type(index).__name__}"
            )
        normalized = int(index)
        if normalized < 0:
            normalized += len(self.samples)
        if normalized < 0 or normalized >= len(self.samples):
            raise IndexError(f"Dataset index out of range: {index}")
        return normalized

    @staticmethod
    def _validate_sparse_ratio(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"sparse_ratio must be a real number, got {type(value).__name__}"
            )
        ratio = float(value)
        if not 0 < ratio <= 1:
            raise ValueError(f"sparse_ratio must be in range (0, 1], got {ratio}")
        return ratio

    @staticmethod
    def _validate_seed(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                f"seed must be a non-negative integer, got {type(value).__name__}"
            )
        seed = int(value)
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        return seed

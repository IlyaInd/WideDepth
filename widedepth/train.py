"""NumPy-first loader for the WideDepth outdoor training dataset."""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, get_args

import cv2
import numpy as np

TrainTask = Literal["stereo", "sparse_depth"]


@dataclass(frozen=True)
class _TrainSamplePaths:
    """Named, immutable modality paths prevent swaps and key-typo failures."""

    sample_id: str
    image: Path
    target: Path
    right_image: Optional[Path] = None


class WideDepthTrainDataset:
    """Access synchronized samples from the WideDepth outdoor train release.

    The root must directly contain the released modality folders. Only folders
    required by the selected task are inspected, and samples are built from
    their numeric PNG-stem intersection. RGB arrays are returned in RGB channel
    order as ``uint8``. Disparity is returned in pixels and sparse depth in
    metres, both as ``float32``. A target value of zero is invalid.

    Args:
        root_dir: Directory containing the released modality folders.
        task: Either ``stereo`` or ``sparse_depth``.
        crop_equirect: Apply the historical equirectangular crop
            ``[449:1665, 994:3106]``. It is available only for ``stereo``.
        rotate_ccw: Rotate every returned image, target, and mask 90 degrees
            counter-clockwise.

    Returns:
        ``stereo`` samples contain ``sample_id``, ``left_image``,
        ``right_image``, ``disparity``, and ``valid_mask``. ``sparse_depth``
        samples contain ``sample_id``, ``image``, ``sparse_depth``, and
        ``valid_mask``.

    Attributes:
        sample_ids: Synchronized sample IDs in deterministic numeric order.
        unmatched_files: Numeric PNG filenames excluded because the same ID is
            absent from at least one modality required by the selected task.
    """

    SUPPORTED_TASKS = frozenset(get_args(TrainTask))

    _DIRECTORIES_BY_TASK = {
        "stereo": ("equirect_up", "equirect_down_virt", "disp_equirect"),
        "sparse_depth": ("left_fisheye", "depth_fisheye"),
    }
    _PRIMARY_DIRECTORY_BY_TASK = {
        "stereo": "equirect_up",
        "sparse_depth": "left_fisheye",
    }
    # Compatibility crop from the original training pipeline. It produces a
    # 1216x2112 view; its geometric derivation is not part of the release.
    _CROP_ROWS = slice(449, 1665)
    _CROP_COLUMNS = slice(994, 3106)
    _DISPARITY_SCALE = 100.0
    _DEPTH_SCALE = 1000.0

    def __init__(
        self,
        root_dir: Union[str, Path],
        task: TrainTask,
        crop_equirect: bool = False,
        rotate_ccw: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.task = task
        self.crop_equirect = crop_equirect
        self.rotate_ccw = rotate_ccw

        self._validate_root()
        self._validate_request()
        self.samples, self.unmatched_files = self._discover_samples()
        self.sample_ids: Tuple[str, ...] = tuple(
            sample.sample_id for sample in self.samples
        )

        if self.unmatched_files:
            counts = ", ".join(
                f"{folder}={len(filenames)}"
                for folder, filenames in self.unmatched_files.items()
            )
            warnings.warn(
                "Ignored numeric PNGs whose IDs are absent from at least one "
                "required modality "
                f"for task={self.task!r}: {counts}",
                RuntimeWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        index = self._normalize_index(index)
        paths = self.samples[index]
        image = self._read_rgb(paths.image)

        if self.task == "stereo":
            if paths.right_image is None:
                raise RuntimeError(
                    f"Internal error: missing stereo path for {paths.sample_id}"
                )
            right_image = self._read_rgb(paths.right_image)
            disparity = self._read_scaled_target(
                paths.target,
                scale=self._DISPARITY_SCALE,
                target_name="disparity",
            )
            self._validate_alignment(paths.sample_id, image, disparity, right_image)
            image, right_image, disparity = self._apply_geometry(
                image,
                right_image,
                disparity,
            )
            return {
                "sample_id": paths.sample_id,
                "left_image": image,
                "right_image": right_image,
                "disparity": disparity,
                "valid_mask": disparity > 0,
            }

        sparse_depth = self._read_scaled_target(
            paths.target,
            scale=self._DEPTH_SCALE,
            target_name="sparse depth",
        )
        self._validate_alignment(paths.sample_id, image, sparse_depth)
        image, sparse_depth = self._apply_geometry(image, sparse_depth)
        return {
            "sample_id": paths.sample_id,
            "image": image,
            "sparse_depth": sparse_depth,
            "valid_mask": sparse_depth > 0,
        }

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"WideDepth train root does not exist: {self.root_dir}"
            )
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"WideDepth train root is not a directory: {self.root_dir}"
            )

    def _validate_request(self) -> None:
        if not isinstance(self.task, str) or self.task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task {self.task!r}; expected one of {sorted(self.SUPPORTED_TASKS)}"
            )
        if not isinstance(self.crop_equirect, bool):
            raise TypeError(
                f"crop_equirect must be a bool, got {type(self.crop_equirect).__name__}"
            )
        if not isinstance(self.rotate_ccw, bool):
            raise TypeError(
                f"rotate_ccw must be a bool, got {type(self.rotate_ccw).__name__}"
            )
        if self.crop_equirect and self.task != "stereo":
            raise ValueError(
                "crop_equirect=True is available only for task='stereo'; "
                "sparse-depth samples use fisheye geometry"
            )

    def _discover_samples(
        self,
    ) -> Tuple[Tuple[_TrainSamplePaths, ...], Dict[str, Tuple[str, ...]]]:
        directory_names = self._DIRECTORIES_BY_TASK[self.task]
        missing = [
            name for name in directory_names if not (self.root_dir / name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required modality directories under {self.root_dir}: {missing}. "
                "Pass the directory containing the modality folders directly."
            )

        files_by_directory = {
            name: self._discover_numeric_pngs(self.root_dir / name)
            for name in directory_names
        }
        common_ids = set.intersection(
            *(set(files_by_id) for files_by_id in files_by_directory.values())
        )
        if not common_ids:
            counts = ", ".join(
                f"{name}={len(files_by_id)}"
                for name, files_by_id in files_by_directory.items()
            )
            raise FileNotFoundError(
                "No numeric PNG ID is shared by all modalities required for "
                f"task={self.task!r} under "
                f"{self.root_dir}; discovered: {counts}"
            )

        unmatched_files = {
            name: tuple(
                files_by_id[numeric_id].name
                for numeric_id in sorted(set(files_by_id) - common_ids)
            )
            for name, files_by_id in files_by_directory.items()
            if set(files_by_id) - common_ids
        }

        primary_name = self._PRIMARY_DIRECTORY_BY_TASK[self.task]
        samples = []
        for numeric_id in sorted(common_ids):
            primary_path = files_by_directory[primary_name][numeric_id]
            if self.task == "stereo":
                samples.append(
                    _TrainSamplePaths(
                        sample_id=primary_path.stem,
                        image=primary_path,
                        right_image=files_by_directory["equirect_down_virt"][
                            numeric_id
                        ],
                        target=files_by_directory["disp_equirect"][numeric_id],
                    )
                )
            else:
                samples.append(
                    _TrainSamplePaths(
                        sample_id=primary_path.stem,
                        image=primary_path,
                        target=files_by_directory["depth_fisheye"][numeric_id],
                    )
                )

        return tuple(samples), unmatched_files

    @staticmethod
    def _discover_numeric_pngs(directory: Path) -> Dict[int, Path]:
        files_by_id: Dict[int, Path] = {}
        for candidate in sorted(directory.iterdir(), key=lambda item: item.name):
            if (
                not candidate.is_file()
                or candidate.suffix.lower() != ".png"
                or not candidate.stem.isdigit()
            ):
                continue
            numeric_id = int(candidate.stem)
            previous = files_by_id.get(numeric_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate numeric PNG ID {numeric_id} in {directory}: "
                    f"{previous.name!r} and {candidate.name!r}"
                )
            files_by_id[numeric_id] = candidate

        if not files_by_id:
            raise FileNotFoundError(f"No numeric PNG files found in {directory}")
        return files_by_id

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
    def _read_scaled_target(path: Path, scale: float, target_name: str) -> np.ndarray:
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None:
            raise OSError(f"Failed to decode {target_name}: {path}")
        if encoded.ndim != 2 or not np.issubdtype(encoded.dtype, np.integer):
            raise ValueError(
                f"Expected a single-channel integer {target_name} at {path}, got "
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
        sample_id: str,
        image: np.ndarray,
        target: np.ndarray,
        right_image: Optional[np.ndarray] = None,
    ) -> None:
        expected_shape = image.shape[:2]
        if target.shape != expected_shape:
            raise ValueError(
                f"Spatial mismatch in sample {sample_id}: image shape={expected_shape}, "
                f"target shape={target.shape}"
            )
        if right_image is not None and right_image.shape[:2] != expected_shape:
            raise ValueError(
                f"Spatial mismatch in sample {sample_id}: left shape={expected_shape}, "
                f"right shape={right_image.shape[:2]}"
            )

    def _apply_geometry(self, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        transformed = arrays
        if self.crop_equirect:
            for array in transformed:
                height, width = array.shape[:2]
                if height < self._CROP_ROWS.stop or width < self._CROP_COLUMNS.stop:
                    raise ValueError(
                        "Cannot apply equirectangular crop "
                        f"[{self._CROP_ROWS.start}:{self._CROP_ROWS.stop}, "
                        f"{self._CROP_COLUMNS.start}:{self._CROP_COLUMNS.stop}] "
                        f"to shape={(height, width)}"
                    )
            transformed = tuple(
                array[self._CROP_ROWS, self._CROP_COLUMNS].copy()
                for array in transformed
            )
        if self.rotate_ccw:
            transformed = tuple(np.rot90(array, k=1).copy() for array in transformed)
        return transformed

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

# WideDepth: Millimeter-Accurate Fisheye Depth Estimation

[Paper](https://arxiv.org/abs/2605.24074) |
[Project page](https://ilyaind.github.io/WideDepth/) |
[Benchmark dataset](https://huggingface.co/datasets/IlyaInd/WideDepth) |
[Training dataset](https://huggingface.co/datasets/IlyaInd/WideDepth-train)

<p align="center">
  <img src="docs/assets/table_1.png" width="900" alt="WideDepth benchmark overview">
</p>

WideDepth provides millimeter-accurate indoor depth and disparity ground truth
for fisheye perception, together with outdoor stereo and sparse LiDAR-derived
training data. This repository contains lightweight, NumPy-first loaders for
both public Hugging Face releases.

## Quickstart notebook

The [Run-All introduction notebook](notebooks/widedepth_quickstart.ipynb)
downloads one small random sample from each Hugging Face dataset, demonstrates
all six public loader tasks, prints their array contracts, and visualizes the
results.

## Installation

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository does not yet provide an installable wheel. Run examples and the
notebook from the repository root so that `widedepth` is importable. PyTorch is
not required.

## Download the datasets

Download and extract either release from Hugging Face:

- [WideDepth benchmark](https://huggingface.co/datasets/IlyaInd/WideDepth):
  indoor evaluation scenes with dense depth and disparity.
- [WideDepth train](https://huggingface.co/datasets/IlyaInd/WideDepth-train):
  outdoor synchronized views, stereo disparity, and projected sparse LiDAR
  depth.

Pass the extracted dataset directory—not this code repository—to the relevant
loader. To download a release from Python, choose a destination with enough
free space:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IlyaInd/WideDepth",  # or "IlyaInd/WideDepth-train"
    repo_type="dataset",
    local_dir="/path/to/WideDepth",
)
```

## Benchmark loader

A benchmark root contains scene directories. One camera configuration has the
following relevant structure:

```text
WideDepth/
└── 001_057_000/
    └── 120FOV/
        └── 65mm/
            ├── CENTER/
            │   ├── pano_crop.png
            │   ├── depth/pano_crop.png
            │   └── disparity/pano_crop.png
            ├── DOWN/pano_crop.png
            └── RIGHT/pano_crop.png
```

Load one task and configuration:

```python
from widedepth import WideDepthBenchmarkDataset

dataset = WideDepthBenchmarkDataset(
    root_dir="/path/to/WideDepth",
    task="mono",
    image_type="pano_crop",
    fov=120,
    baseline_mm=65,
)

sample = dataset[0]
print(sample["sample_id"], sample["image"].shape, sample["depth"].shape)
```

Supported FOVs are `120`, `140`, `165`, and `195` degrees. Supported baselines
are `20`, `65`, `120`, `200`, and `300` mm. FOV 120 provides `fisheye`, `pano`,
`pano_crop`, `pinhole`, and `pinhole_crop`; the wider FOVs provide `fisheye`,
`pano`, and `pano_crop`. Fisheye disparity is not released, so fisheye stereo
requests are rejected.

## Training loader

The training root directly contains its synchronized modality folders:

```text
WideDepth_train/
├── left_fisheye/
├── right_fisheye/
├── equirect_up/
├── equirect_down_virt/
├── disp_equirect/
└── depth_fisheye/
```

Load stereo training data:

```python
from widedepth import WideDepthTrainDataset

dataset = WideDepthTrainDataset(
    root_dir="/path/to/WideDepth_train",
    task="stereo",
    crop_equirect=False,
    rotate_ccw=False,
)

sample = dataset[0]
print(sample["sample_id"], sample["left_image"].shape, sample["disparity"].shape)
```

The loader intersects numeric PNG IDs only across the folders needed by the
selected task. It reports files outside that intersection in
`dataset.unmatched_files` and emits one warning when any are excluded. Native
geometry is preserved by default. `crop_equirect=True` applies the historical
`[449:1665, 994:3106]` stereo crop, and `rotate_ccw=True` rotates all returned
arrays counter-clockwise.

## Tasks and returned arrays

| Dataset | Task | Returned fields |
|---|---|---|
| Benchmark | `mono` | `sample_id`, `image`, `depth`, `valid_mask` |
| Benchmark | `depth_completion` | `sample_id`, `image`, `sparse_depth`, `depth`, `valid_mask` |
| Benchmark | `stereo_vertical` | `sample_id`, `left_image`, `right_image`, `disparity`, `valid_mask` |
| Benchmark | `stereo_horizontal` | `sample_id`, `left_image`, `right_image`, `disparity`, `valid_mask` |
| Train | `stereo` | `sample_id`, `left_image`, `right_image`, `disparity`, `valid_mask` |
| Train | `sparse_depth` | `sample_id`, `image`, `sparse_depth`, `valid_mask` |

All RGB images are HWC `uint8` arrays in RGB channel order. Depth and sparse
depth are `float32` metres; disparity is `float32` pixels; `valid_mask` is a
Boolean array. Encoded zero target values are invalid.

Benchmark depth completion simulates a reproducible sparse input from dense
benchmark ground truth. Configure it with `sparse_ratio` and `seed`; the dense
`depth` remains available as supervision. Train `sparse_depth` instead returns
the released LiDAR-derived sparse measurements and has no aligned dense-depth
target.

Benchmark scenes missing files required by a requested configuration are
omitted with a `RuntimeWarning` and listed in `dataset.skipped_scenes`. The
loader never synthesizes or substitutes a missing released image.

## Citation

If WideDepth is useful in your research, please cite:

```bibtex
@article{indyk2026widedepth,
  title   = {WideDepth: Millimeter-Accurate Benchmark for Fisheye Depth Estimation},
  author  = {Indyk, Ilia and Penshin, Ignat and Sosin, Ivan and Monastyrny, Maxim and Valenkov, Aleksei and Makarov, Ilya},
  journal = {arXiv preprint arXiv:2605.24074},
  year    = {2026}
}
```

## License

The code license has not been selected yet. Until a license is added, the code
is all rights reserved. Dataset licenses and usage terms are separate from the
code and are defined on the respective Hugging Face dataset pages linked above.

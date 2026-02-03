# Multi-task Guided OIQA (CVIQ)

This repo starts a reproducible baseline for the CVIQ experiments described in
"Multi-task Guided No-Reference Omnidirectional Image Quality Assessment with Feature Interaction".
The current implementation focuses on dataset loading, a multi-task training loop, and a
baseline model that mirrors the local/global fusion + auxiliary heads described in the paper.

## Dataset layout

```
CVIQ/
  001.png
  ...
  544.png

view_ports/
  AVC/
    001_fov1.png
    ...
    001_fov20.png
  HEVC/
  JPEG/
  ref/
```

## Download CVIQ (optional)

```bash
python -m pip install gdown
mkdir -p data
gdown --id 12E-sDZOq0DfCtNNwdyer7azfLZNNva6N -O data/CVIQ.zip
unzip -q data/CVIQ.zip -d data
```

> Note: this download is only for local validation in the Codex environment. If you already
> have CVIQ prepared in your own `data/` structure, you can skip this section.

Then verify the dataset layout (including the viewport folders):

```bash
python scripts/verify_cviq.py --data-root data/CVIQ --viewports-root data/view_ports
```

## Viewport tool (MATLAB)

The `twentyviewportstool.zip` file (from the main branch upload) can be unpacked with:

```bash
python scripts/unpack_viewport_tool.py --zip-path twentyviewportstool.zip
```

The extracted MATLAB scripts will be placed under `tools/twenty_viewports/`.

## Annotation CSV

Create `data/cviq_annotations.csv` with at least the following columns:

| column | description |
| --- | --- |
| image_id | image ID, e.g. `1` or `001` |
| mos | mean opinion score |
| compression_type | `JPEG`, `AVC`, or `HEVC` |
| distortion_level | integer label for compression strength |

Example row:

```
image_id,mos,compression_type,distortion_level
1,62.5,HEVC,3
```

## Train

```bash
python -m mtg_oiqa.train \
  --annotations data/cviq_annotations.csv \
  --data-root data/CVIQ \
  --viewports-root data/view_ports
```

The training script reports PLCC/SRCC/RMSE using the five-parameter nonlinear mapping
described in the paper.

## Notes

- The current model uses a ResNet50 backbone as a placeholder for the global branch (VMamba
  is not yet integrated).
- The reproduction now includes a bidirectional pseudo-reference module and a multi-scale
  bi-stream fusion head (BS-MSFA) to more closely match the paper. These remain lightweight
  CNN-based versions to keep the baseline runnable while we iterate.

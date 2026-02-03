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

## Notes

- The current model uses ResNet50 backbones as a placeholder for the global branch (VMamba
  is not yet integrated).
- Bidirectional pseudo-reference and BS-MSFA fusion are simplified to a mean aggregation and
  a shared MLP fusion head. These can be replaced as we expand the reproduction.

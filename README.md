# Vision Pipeline Dataset Cleaner

Automated pipeline curating 1,147 raw images → 170 high-quality person crops with 97.65% accuracy.

## Quick Start

```
conda create -n dev python=3.13 && conda activate dev
```
```
make install   # install dependencies
make run       # end-to-end pipeline
make evaluate  # validate results
```

## Pipeline Overview

| Phase | Method | Input → Output |
|-------|--------|---|
| 1 | CleanVision (remove duplicates) | 1,147 → 1,001 |
| 2 | YOLOv8s (person detection) | 1,001 → 536 |
| 3 | YOLOv8-Pose (full-body detection) | 536 → 202 |
| 4 | InsightFace (face and age detection) | 202 → 170 |
| 5 | EasyOCR (ad detection) | 170 → 170 |

## System Design

### Architecture

**Batch-style filtering:** Each filter processes ALL remaining images sequentially (no intermediate folders):
1. Call `setup()` once per filter for initialization
2. For each image, call `apply(image_path) → FilterResult(keep: bool)`
3. Pass only kept images to next filter
4. Copy final results to output

Note: In-memory processing is efficient and avoids redundant file I/O.

## Phase-by-Phase Design Decisions Summary
> For detailed experimental results and explanation, please refer to individual notebooks. 
#### Phase 1: Duplicate Detection (CleanVision)
- Method: Identify and remove exact duplicates
- Decision: Duplicate images increase filtering time for subsequent phases. Keep low-quality images since they may be valid in later phases
- Notebook: `notebooks/01_data_quality.ipynb`

#### Phase 2: Person Detection (YOLOv8s)
- Method: Pre-trained YOLOv8s detecting person class (confidence ≥ 0.5)
- Decision: Fast, no fine-tuning needed, standard object detector
- Notebook: `notebooks/02_person_detection.ipynb`

#### Phase 3: Full-Body Validation (YOLOv8-Pose)
- Method: Keypoint detection requiring visible head (nose) AND legs (knees/ankles)
- Decision: Captures "full-body" requirement without custom model training
- Notebook: `notebooks/03_fullbody_validation.ipynb`

#### Phase 4: Age Filtering (InsightFace)
- Method: Face detection + InsightFace age estimation, keep age ≥ 13
- Decision: Face detection using InsightFace is more accurate than MediaPipe. Single inference handles both detection and age estimation, pre-trained on large datasets
- Notebook: `notebooks/04_age_estimation.ipynb`

#### Phase 5: Advertisement Detection (EasyOCR)
- Method: OCR text extraction + keyword matching (sale, offer, discount, %, $, €, £)
- Decision: CLIP method is inaccurate because it's not trained on advertisements. Rule-based OCR approach generalizes across datasets without requiring training
- Notebook: `notebooks/05_advertisement_detection.ipynb`

## Project Structure
```
.
├── main.py              # CLI entry point
├── evaluate.py          # Validation script
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── src/
│   ├── base.py          # BaseFilter, FilterResult
│   ├── runner.py        # PipelineRunner
│   ├── dedupe.py        # Phase 1
│   ├── person_detector.py   # Phase 2
│   ├── fullbody_filter.py   # Phase 3
│   ├── age_filter.py        # Phase 4
│   └── advertisement_filter.py  # Phase 5
├── notebooks/           # Experimentation notebooks
└── data/
    ├── original_raw/    # Input
    └── final/           # Output
```

## Results

- Retention: 14.8% of original
- Manual validation:** 4 errors / 170 images → 97.65% accuracy
- Performance (1,147 images on CPU, ~90 seconds total):
    - Phase 1 (Dedupe): <1s
    - Phase 2 (Person Detection): ~42s (25 imgs/sec)
    - Phase 3 (Full-Body): ~15s (36 imgs/sec)
    - Phase 4 (Age Filter): ~29s (7 imgs/sec) — **bottleneck**
    - Phase 5 (Ad Detection): included in Phase 4

Known Errors:
- 1 child not detected (age detection edge case)
- 2 mannequins missed (no face detected)
- 1 ad with minimal text (OCR limitation)

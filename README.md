# Vision Pipeline Dataset Cleaner

Automated pipeline curating 1,147 raw images → 171 high-quality person crops.

## Quick Start

### 1. Environment Setup
```bash
conda create -n dev python=3.13 && conda activate dev
make install
```

### 2. Download Dataset
Download the image dataset from Google Drive to `data/original_raw/`:
```bash
# Create data directory
mkdir -p data/original_raw

# Download dataset (replace GOOGLE_DRIVE_FILE_ID with actual ID)
pip install gdown
gdown --folder https://drive.google.com/drive/folders/GOOGLE_DRIVE_FILE_ID -O data/original_raw/

# Or download manually and extract to data/original_raw/
```

### 3. Run Pipeline
```bash
make run       # Execute full pipeline
make evaluate  # Validate results
```

**Expected output:** 168 cleaned images in `data/final/` with 97.62% accuracy

## Pipeline Overview

| Phase | Method | Input → Output |
|-------|--------|---|
| 1 | imagededup PHash (remove duplicates) | 1,147 → 999 |
| 2 | Person Detection (Removed) | — |
| 3 | YOLOv8-Pose (full-body detection) | 999 → 227 |
| 4 | InsightFace (face and age detection) | 227 → 172 |
| 5 | CLIP (ad detection) | 172 → 171 |

> **Note:** Phase 2 (Person Detection) was removed as FullBodyFilter now implicitly handles person detection.

## Metrics Definition
- Positive (P) = Valid image that should be kept (person, full-body, age ≥ 13, no ads)
- Negative (N) = Invalid image that should be filtered (duplicate, not-person, incomplete body, age < 13, has ads)
- Confusion Matrix:
    | | **Predicted KEEP** | **Predicted FILTER** |
    |---|---|---|
    | **Actually Valid (P)** | TP (correct keep) | FN (over-filtered) |
    | **Actually Invalid (N)** | FP (wrong keep) | TN (correct filter) |

Performance Metrics:
- True Positive (TP): Valid image correctly kept
- True Negative (TN): Invalid image correctly filtered
- False Positive (FP): Invalid image incorrectly kept — contaminates final dataset
- False Negative (FN): Valid image incorrectly filtered — loses good data
- Accuracy = (TP + TN) / (TP + TN + FP + FN)

In this project, we prioritise precision:<br>
- Precision = TP / (TP + FP) = "Of the images we kept, how many are actually valid?"
- invalid data in final set is worse than over-filtering

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
#### Phase 1: Duplicate Detection (imagededup PHash)
- Method: Perceptual hashing to identify duplicate and near-duplicate images
- Decision: Prioritize scalability and faster duplicate mining for larger datasets. Keep low-quality images since they may be valid in later phases
- Notebook: `notebooks/01_data_quality.ipynb`

#### Phase 2: Person Detection (YOLOv8s) - **Removed**
- **Note:** Person detection was found to be redundant. FullBodyFilter now implicitly filters out images without people.

#### Phase 3: Full-Body Validation (YOLOv8-Pose)
- Method: Keypoint detection requiring visible head (nose) AND legs (knees/ankles)
- Decision: Captures "full-body" requirement without custom model training
- Notebook: `notebooks/03_fullbody_validation.ipynb`

#### Phase 4: Age Filtering (InsightFace)
- Method: Face detection + InsightFace age estimation, keep age ≥ 13
- Decision: Face detection using InsightFace is more accurate than MediaPipe. Single inference handles both detection and age estimation, pre-trained on large datasets
- Notebook: `notebooks/04_age_estimation.ipynb`

#### Phase 5: Advertisement Detection (CLIP)
- Method: CLIP visual-semantic matching with prompts: "promotional advertisement or marketing image" vs "candid photo of a person"
- Decision: CLIP chosen over EasyOCR (50% vs 0% recall on known ads). OCR failed on brand-only/text-free ads. Revised prompts reduced false positives from 65% to 0.6% while maintaining recall. Trade-off: 5x slower but critical for precision
- Notebook: `notebooks/05_advertisement_detection.ipynb`

## Project Structure
```
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
- Retention: 14.9% of original (171/1,147)
- Manual validation: 171 images in final set
- Performance (1,147 images on CPU, ~51 seconds total):
    - Phase 1 (Dedupe): <1s (999 images)
    - Phase 2: Removed
    - Phase 3 (Full-Body): 27.43s (227 images)
    - Phase 4 (Age Filter): 13.24s (172 images)
    - Phase 5 (Ad Detection): 10.60s (171 images)
    - **Total time taken:** 51.27s

### Note on Pipeline Change
- When Phase 2 (Person Detection) was included, images flowed from Phase 1 (999) → Phase 2 (person detection) (534) → Phase 3 (full-body) (201). - After removing Phase 2, all 999 deduplicated images go directly to full-body filtering, resulting in 227 images passing Phase 3. This means FullBodyFilter now handles both person and full-body detection, and more images are evaluated for full-body presence.

> **Summary:** With Phase 2 removed, Phase 1 to 3 now goes from 999 → 227 images (previously 999 → 534 → 201), simplifying the pipeline and relying on FullBodyFilter for both person and full-body filtering.

> **Note:** Person Detection phase was removed; FullBodyFilter now handles person detection.

Known Errors:
- False Positives (FP): 2 invalid images kept (1 child passed age filter, 1 ad passed ad detector)
- False Negatives (FN): 2 valid images discarded (2 mannequins filtered by full-body validator)
- Risk Priority: FP > FN (invalid data in final set is worse than over-filtering)


## Future development
- To improve on development:
    - use mlflow for experimental tracking
    - use dvc for dataset tracking
    - use logging for debugging
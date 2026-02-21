# Vision Pipeline Dataset Cleaner

Automated pipeline curating 1,147 raw images to a final set of 171 high-quality, full-body images of people, with most advertisements and mannequins filtered out. A small number of ads or mannequins may remain due to pipeline limitations. The pipeline achieves 96.4% precision on the final, curated dataset.

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

**Expected output:** 168–171 cleaned images in `data/final/` with 96.4% precision and 97.6% accuracy

## Pipeline Overview

| Phase | Method | Input → Output | Precision |
|-------|--------|---|-----------|
| 1 | imagededup PHash (remove duplicates) | 1,147 → 999 | - |
| 2 | Person Detection (Removed) | — | — |
| 3 | YOLOv8-Pose (full-body detection) | 999 → 227 | 85.4% |
| 4 | InsightFace (face and age detection) | 227 → 172 | 95.93% |
| 5 | CLIP (ad detection) | 172 → 171 | 96.4% |

> **Note:** Phase 2 (Person Detection) was removed as FullBodyFilter now implicitly handles person detection.

## Metrics Definition
- **Positive (P):** Valid image to keep (person, full-body, age ≥ 13, no ads)
- **Negative (N):** Invalid image to filter (duplicate, not-person, incomplete body, age < 13, has ads)
- **Confusion Matrix:**
    - TP: Valid image kept
    - FP: Invalid image kept (contaminates dataset)
    - TN: Invalid image filtered
    - FN: Valid image filtered (lost data)
- **Key metrics:**
    - Precision = TP / (TP + FP)  (priority: minimize FP)
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - We prioritize precision: better to over-filter than keep invalid data.

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
- Decision: CLIP chosen over EasyOCR after experiments on an expanded ad set (n=30):
    - CLIP achieved 100% detection rate (30/30 ads detected), including brand-only and visual ads
    - OCR detected only 40% (12/30), missing brand-only/text-free ads
    - Prompt engineering was key to reducing false positives and improving robustness
    - Precision is prioritized: CLIP minimizes invalid images in the final set
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
- Final precision: **96.4%** (on 1,147-image dataset)
- Performance (1,147 images):
    - Phase 1 (Dedupe): <1s (999 images)
    - Phase 2: Removed
    - Phase 3 (Full-Body): 27.43s (227 images)
    - Phase 4 (Age Filter): 13.24s (172 images)
    - Phase 5 (Ad Detection): 10.60s (171 images)
    - **Total time taken:** 51.27s


### Note on Pipeline Change
Previously, Phase 2 (Person Detection) filtered images before full-body validation, resulting in: 999 (deduped) → 534 (person detected) → 201 (full-body).

Now, with Phase 2 removed, all 999 deduplicated images go directly to full-body filtering: 999 → 227. FullBodyFilter now handles both person and full-body detection, simplifying the pipeline and increasing the number of images evaluated for full-body presence.

> **Summary:** Removing Phase 2 streamlines the pipeline and relies on FullBodyFilter for both person and full-body filtering.


## Future development
- To improve on development:
    - use mlflow for experimental tracking
    - use dvc for dataset tracking
    - use logging for debugging
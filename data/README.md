# data/

This folder stores all input, output, and intermediate images for the pipeline.

- `original_raw/` — Place the original dataset here (input images)
- `cleaned_raw/` — After deduplication (Phase 1)
- `person_only/` — After person detection (Phase 2)
- `fullbody_filtered/` — After full-body validation (Phase 3)
- `age_filtered/` — After age filtering (Phase 4)
- `ad_filtered/` — After advertisement filtering (Phase 5, intermediate)
- `final/` — Final cleaned images (output)
- `notebook/` — Notebook-specific outputs (e.g., visualizations, experiments)

Folders are created automatically by the pipeline and notebooks as needed.

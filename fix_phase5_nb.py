#!/usr/bin/env python3
"""Script to clean up Phase 5 notebook - remove OCR, keep only CLIP"""

import json

# Read the notebook
with open('notebooks/05_advertisement_detection.ipynb', 'r') as f:
    nb = json.load(f)

# Cells to keep (by index) after modifications
# We'll modify specific cells and remove debug cells

# Cell modifications
modifications = {
    6: {  # Import cell
        'source': [
            "import cv2\n",
            "import numpy as np\n",
            "import torch\n",
            "from PIL import Image\n",
            "\n",
            "# Install open-clip-torch if needed\n",
            "try:\n",
            "    import open_clip\n",
            "except ImportError:\n",
            "    print(\"Installing open-clip-torch...\")\n",
            "    !pip install open-clip-torch\n",
            "    import open_clip\n",
            "\n",
            "print(\"✓ Loaded open-clip\")"
        ]
    },
    8: {  # CLIP model loading
        'source': [
            "# Load CLIP model and config settings\n",
            "clip_model_name = config['advertisement']['clip_model']\n",
            "clip_margin = config['advertisement']['clip_margin']\n",
            "\n",
            "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "print(f\"Using device: {device}\")\n",
            "\n",
            "model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', device=device)\n",
            "model.eval()\n",
            "\n",
            "# Define classification prompts\n",
            "ad_prompts = [\n",
            "    \"a person in an advertisement\",\n",
            "    \"a person in a store or shop display\",\n",
            "    \"a person wearing a product advertisement\",\n",
            "    \"a person on a poster or billboard\",\n",
            "]\n",
            "\n",
            "natural_prompts = [\n",
            "    \"a natural photo of a person\",\n",
            "    \"a candid photo of a person\",\n",
            "    \"a portrait of a person\",\n",
            "    \"a person in a natural setting\"\n",
            "]\n",
            "\n",
            "print(f\"✓ Loaded CLIP model ({clip_model_name})\")\n",
            "print(f\"  Ad prompts: {len(ad_prompts)}\")\n",
            "print(f\"  Natural prompts: {len(natural_prompts)}\")\n",
            "print(f\"  Margin threshold: {clip_margin}\")"
        ]
    }
}

# Cells to remove (indices)
cells_to_remove = [10, 11, 16, 17, 18, 19]  # OCR reader, OCR function, debug cells

# Apply modifications
for idx, content in modifications.items():
    if idx < len(nb['cells']):
        nb['cells'][idx]['source'] = content['source']

# Remove cells (in reverse order to maintain indices)
for idx in sorted(cells_to_remove, reverse=True):
    if idx < len(nb['cells']):
        del nb['cells'][idx]

# Write back
with open('notebooks/05_advertisement_detection_clean.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Created cleaned notebook: notebooks/05_advertisement_detection_clean.ipynb")
print("Review and rename to replace the original if correct.")

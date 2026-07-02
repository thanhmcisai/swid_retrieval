#!/usr/bin/env bash
# Install dependencies for swid_retrieval (Colab). Run once per session.
set -euo pipefail

pip -q install torch torchvision timm scikit-learn tqdm albumentations opencv-python-headless
pip -q install git+https://github.com/openai/CLIP.git
echo "✅ setup done"

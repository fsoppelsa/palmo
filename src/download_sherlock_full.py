#!/usr/bin/env python3
"""
Script for downloading pre-trained checkpoints from the HuggingFace Hub.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing Course
Academic Year 2025/2026
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi, login

# HuggingFace repository
REPO_ID = "Fabrizio99/unipa-palmo-nlp"

# Files to download
FILES = [
    "palmo-sherlock-best.pt",
    "palmo-sherlock-pretrained.pt",
    "palmo-sherlock-quantized.pt",
    "palmo-sherlock-tuned.pt",
    "palmo-sherlock-tuned-best.pt",
    "palmo-sherlock-tuned-quantized.pt",
    "palmo-bohemia-best.pt",
    "palmo-bohemia-pretrained.pt",
    "palmo-bohemia-quantized.pt",
    "palmo-bohemia-tuned.pt",
    "palmo-bohemia-tuned-best.pt",
]

if Path("hf_token").exists():
    HF_TOKEN = Path("hf_token").read_text().strip()
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Autenticato con token!")
else:
    print("Token non trovato in hf_token")

api = HfApi()

# Local directory to save files
CHECKPOINT_DIR = "checkpoints"

def main():
    # Create checkpoints directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for filename in FILES:
        print(f"=> Download{filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=CHECKPOINT_DIR,
                local_dir_use_symlinks=False
            )
            print(f"=> Saved to: {downloaded_path}\n")
        except Exception as e:
            print(f"=> Error downloading {filename}: {e}\n")
    
    print("Done!")


if __name__ == "__main__":
    main()

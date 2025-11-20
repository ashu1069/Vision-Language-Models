#!/usr/bin/env python3
"""
Script to download PaliGemma model weights from Hugging Face.
You need to be logged in to Hugging Face and have accepted the license terms.
"""

import os
from huggingface_hub import snapshot_download, login, whoami

def download_model():
    model_id = "google/paligemma-3b-pt-224"
    local_dir = "/home/stu12/s11/ak1825/Projects/Vision-Language-Models/PaliGemma/models/paligemma-3b-pt-224"
    
    # Check if user is logged in
    try:
        user_info = whoami()
        print(f"✓ Logged in as: {user_info.get('name', 'Unknown')}")
    except Exception:
        print("You are not logged in to Hugging Face.")
        print("Please provide your Hugging Face token.")
        print("You can get it from: https://huggingface.co/settings/tokens")
        print()
        token = input("Enter your Hugging Face token: ").strip()
        if token:
            login(token=token)
            print("✓ Logged in successfully!")
        else:
            print("✗ No token provided. Exiting.")
            return
    
    print(f"Downloading {model_id} to {local_dir}...")
    print("Note: Make sure you have accepted the license terms at:")
    print("      https://huggingface.co/google/paligemma-3b-pt-224")
    print()
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
        )
        print(f"\n✓ Model downloaded successfully to {local_dir}")
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've accepted the license terms at:")
        print("   https://huggingface.co/google/paligemma-3b-pt-224")
        print("2. Check that your token has the correct permissions")
        raise

if __name__ == "__main__":
    download_model()


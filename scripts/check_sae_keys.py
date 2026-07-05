import torch
import os

def check_sae_keys(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading SAE checkpoint from: {file_path}")
    try:
        # Load on CPU to avoid CUDA issues
        sae_dict = torch.load(file_path, map_location="cpu")
        
        print("\n--- Dictionary Keys ---")
        print(list(sae_dict.keys()))
        
        print("\n--- Tensor Shapes ---")
        for k, v in sae_dict.items():
            if hasattr(v, "shape"):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: (Not a tensor, type: {type(v)})")
                
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    # Test with layer 20 as discussed
    target_path = "/home/eagle0914/medical_mi/checkpoints/sae/qwen3.5-9b/layer20.sae.pt"
    check_sae_keys(target_path)

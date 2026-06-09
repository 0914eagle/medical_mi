import torch
import numpy as np
from scipy import stats
import json
import os

# --- Config ---
RESULTS_DIR = "results/features"
TARGET_LAYERS = [15, 20, 25]
TOP_K = 50

def find_ignorance_features(correct_features, ignorance_features, layer_idx, top_k=TOP_K):
    """
    Identify features that are suppressed during ignorance (wrong confident)
    compared to knowledge (correct confident).
    """
    correct_mean = correct_features.mean(0)
    ignorance_mean = ignorance_features.mean(0)
    
    # Difference: high means more active in correct, suppressed in ignorance
    mean_diff = correct_mean - ignorance_mean
    
    t_stats = []
    p_values = []
    
    for feature_idx in range(correct_features.shape[1]):
        c_vals = correct_features[:, feature_idx].numpy()
        i_vals = ignorance_features[:, feature_idx].numpy()
        
        if c_vals.sum() == 0 and i_vals.sum() == 0:
            t_stats.append(0)
            p_values.append(1.0)
            continue
            
        try:
            t, p = stats.ttest_ind(c_vals, i_vals)
            t_stats.append(t if not np.isnan(t) else 0)
            p_values.append(p if not np.isnan(p) else 1.0)
        except:
            t_stats.append(0)
            p_values.append(1.0)
            
    t_stats = torch.tensor(t_stats)
    p_values = torch.tensor(p_values)
    
    # Sort by mean difference
    top_indices = torch.topk(mean_diff, k=top_k).indices
    
    results = {
        "top_feature_indices": top_indices.tolist(),
        "mean_diff_scores": mean_diff[top_indices].tolist(),
        "t_stats": t_stats[top_indices].tolist(),
        "p_values": p_values[top_indices].tolist(),
        "correct_mean_activation": correct_mean[top_indices].tolist(),
        "ignorance_mean_activation": ignorance_mean[top_indices].tolist()
    }
    
    print(f"\nLayer {layer_idx} Top-10 Ignorance Feature Candidates:")
    for i in range(min(10, top_k)):
        idx = top_indices[i].item()
        diff = mean_diff[idx].item()
        t = t_stats[idx].item()
        p = p_values[idx].item()
        print(f"  Feature #{idx:6d}: diff={diff:.4f}, t={t:.2f}, p={p:.4f}")
        
    return results

def main():
    all_layer_results = {}
    
    for layer_idx in TARGET_LAYERS:
        c_path = f"{RESULTS_DIR}/correct_confident_layer{layer_idx}.pt"
        i_path = f"{RESULTS_DIR}/wrong_confident_layer{layer_idx}.pt"
        
        if not os.path.exists(c_path) or not os.path.exists(i_path):
            print(f"Feature files for layer {layer_idx} not found. Skipping.")
            continue
            
        correct_features = torch.load(c_path)
        ignorance_features = torch.load(i_path)
        
        results = find_ignorance_features(correct_features, ignorance_features, layer_idx)
        all_layer_results[layer_idx] = results
        
    output_path = f"{RESULTS_DIR}/ignorance_feature_candidates.json"
    with open(output_path, "w") as f:
        json.dump(all_layer_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

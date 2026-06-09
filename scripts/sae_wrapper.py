import torch
import torch.nn as nn
import os

class SAEWrapper(nn.Module):
    """
    Unified wrapper for different SAE architectures (Qwen-Scope TopK vs Gemma Scope 2 JumpReLU)
    """
    def __init__(self, sae_dict, suite="qwen"):
        super().__init__()
        self.suite = suite
        
        if suite == "qwen":
            # Qwen-Scope (TopK)
            self.W_enc = sae_dict.get("W_enc", sae_dict.get("encoder.weight")).float()
            self.b_enc = sae_dict.get("b_enc", sae_dict.get("encoder.bias")).float()
            self.W_dec = sae_dict.get("W_dec", sae_dict.get("decoder.weight")).float()
            self.b_dec = sae_dict.get("b_dec", sae_dict.get("decoder.bias", None))
            
            # Shape normalization to [d_model, d_sae]
            if self.W_enc.shape[0] != 4096 and self.W_enc.shape[1] == 4096:
                self.W_enc = self.W_enc.t().contiguous()
            if self.W_dec.shape[1] != 4096 and self.W_dec.shape[0] == 4096:
                self.W_dec = self.W_dec.t().contiguous()
                
            self.d_sae = self.W_enc.shape[1]
            self.k = 50 # Default TopK for Qwen-Scope
            
        elif suite == "gemma":
            # Gemma Scope 2 (JumpReLU / Standard)
            # Keys might vary based on the exact version, following common gemma-scope patterns
            self.W_enc = sae_dict.get("W_enc", sae_dict.get("weight")).float()
            self.b_enc = sae_dict.get("b_enc", sae_dict.get("bias")).float()
            self.W_dec = sae_dict.get("W_dec", sae_dict.get("weight_dec", self.W_enc.t())).float()
            self.b_dec = sae_dict.get("b_dec", sae_dict.get("bias_dec", None))
            self.threshold = sae_dict.get("threshold", 0.0) # JumpReLU threshold
            
            self.d_sae = self.W_enc.shape[1]

    def to(self, device):
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        if self.b_dec is not None:
            self.b_dec = self.b_dec.to(device)
        return self

    def encode(self, x):
        """
        Input x: [..., d_model]
        Output: [..., d_sae]
        """
        device = x.device
        x_float = x.float()
        
        if self.b_dec is not None:
            x_float = x_float - self.b_dec.to(device)
            
        pre_act = x_float @ self.W_enc.to(device) + self.b_enc.to(device)
        
        if self.suite == "qwen":
            # TopK implementation
            vals, idxs = torch.topk(pre_act, self.k, dim=-1)
            features = torch.zeros_like(pre_act).scatter_(-1, idxs, torch.relu(vals))
        else:
            # JumpReLU or ReLU implementation
            # For JumpReLU, features are 0 if below threshold
            features = torch.relu(pre_act)
            if hasattr(self, "threshold"):
                features = features * (features > self.threshold).float()
                
        return features

    def decode(self, features):
        """
        Input features: [..., d_sae]
        Output: [..., d_model]
        """
        device = features.device
        recon = features @ self.W_dec.to(device)
        if self.b_dec is not None:
            recon = recon + self.b_dec.to(device)
        return recon

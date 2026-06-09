import torch
import torch.nn as nn

class SAEWrapper(nn.Module):
    """
    Qwen-Scope 및 Gemma-Scope를 위한 최적화된 SAE 래퍼.
    사용자 확인 결과: 
    Qwen-Scope weights: W_enc[65536, 4096], W_dec[4096, 65536]
    """
    def __init__(self, sae_dict, suite="qwen"):
        super().__init__()
        self.suite = suite
        
        # 1. 가중치 로드 (확인된 키: W_enc, W_dec, b_enc, b_dec)
        self.W_enc = sae_dict["W_enc"].float()
        self.b_enc = sae_dict["b_enc"].float()
        self.W_dec = sae_dict["W_dec"].float()
        self.b_dec = sae_dict["b_dec"].float()
        
        # 2. Shape 정규화 (Internal format: W_enc[d_model, d_sae], W_dec[d_sae, d_model])
        # 연산 효율을 위해 x @ W 형태가 되도록 미리 transpose
        
        # W_enc가 [d_sae, d_model]인 경우 -> [d_model, d_sae]로 변경
        if self.W_enc.shape[0] > self.W_enc.shape[1]: 
            self.W_enc = self.W_enc.t().contiguous()
            
        self.d_model = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]
        
        # W_dec가 [d_model, d_sae]인 경우 -> [d_sae, d_model]로 변경
        if self.W_dec.shape[0] != self.d_sae:
            self.W_dec = self.W_dec.t().contiguous()
            
        if suite == "qwen":
            self.k = 50  # Qwen-Scope 기본 TopK
        else:
            self.threshold = sae_dict.get("threshold", 0.0) # JumpReLU용

    def to(self, device):
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)
        return self

    def encode(self, x):
        """
        Input x: [..., d_model]
        Output: [..., d_sae]
        """
        device = x.device
        x_float = x.float()
        
        # Pre-encoder bias subtraction (Qwen-Scope standard)
        x_cent = x_float - self.b_dec.to(device)
            
        pre_act = x_cent @ self.W_enc.to(device) + self.b_enc.to(device)
        
        if self.suite == "qwen":
            # TopK activation
            vals, idxs = torch.topk(pre_act, self.k, dim=-1)
            features = torch.zeros_like(pre_act).scatter_(-1, idxs, torch.relu(vals))
        else:
            # JumpReLU or ReLU activation
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
        recon = (features @ self.W_dec.to(device)) + self.b_dec.to(device)
        return recon

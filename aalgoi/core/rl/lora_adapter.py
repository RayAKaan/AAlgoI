from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.scale = alpha / rank

        d_in = base_layer.in_features
        d_out = base_layer.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_out + lora_out


class LoRAAdapter:

    def __init__(self, network: nn.Module, rank: int = 4, alpha: float = 1.0):
        self.network = network
        self.rank = rank
        self.alpha = alpha
        self._applied = False

    def apply(self):
        if self._applied:
            return
        self.network.query_proj = LoRALinear(
            self.network.query_proj, self.rank, self.alpha
        )
        self.network.key_proj = LoRALinear(
            self.network.key_proj, self.rank, self.alpha
        )
        self._applied = True

    def freeze_base(self):
        for name, param in self.network.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False

    def parameters(self):
        params = []
        for module in [
            self.network.query_proj,
            self.network.key_proj,
        ]:
            if isinstance(module, LoRALinear):
                params.extend([module.lora_A, module.lora_B])
        return params

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for name, module in [
            ('query_proj', self.network.query_proj),
            ('key_proj', self.network.key_proj),
        ]:
            if isinstance(module, LoRALinear):
                state[f'{name}.lora_A'] = module.lora_A.data.clone()
                state[f'{name}.lora_B'] = module.lora_B.data.clone()
        torch.save({
            'lora_state': state,
            'rank': self.rank,
            'alpha': self.alpha,
        }, path)

    def load(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            state = data['lora_state']
            for name, module in [
                ('query_proj', self.network.query_proj),
                ('key_proj', self.network.key_proj),
            ]:
                if isinstance(module, LoRALinear):
                    module.lora_A.data = state[f'{name}.lora_A'].to(module.lora_A.device)
                    module.lora_B.data = state[f'{name}.lora_B'].to(module.lora_B.device)
            return True
        except Exception as e:
            print(f"[LoRAAdapter] Load failed: {e}")
            return False

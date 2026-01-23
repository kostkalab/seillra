from .get_models import get_sei_head_llra, get_sei_projection, get_sei_trunk
import torch.nn as nn
import torch
# import os, sys
# print(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../seimodel-dev/dist/seimodel/src')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../seimodel-dev/dist/seimodel/src')))
# import sei_projection as sm

import os, sys
from typing import Optional, Literal

# here = os.path.dirname(os.path.abspath(__file__))
# seimodel_root = os.path.abspath(os.path.join(here, '../../../../seimodel-dev/dist'))
# sys.path.append(seimodel_root)

# from seimodel.src import get_seimodels as sm
class Sei_LLRA(nn.Module):
    def __init__(self, k: int | None, projection: bool = True, mode: Literal["sequence", "variant"] = "sequence", quant: Literal["CPU", "GPU_fp16", "GPU_int8", None] = None, compile: bool = False):
        super().__init__()
        self.quant = quant
        self.compile = compile
        if self.quant == "CPU":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.quant = "CPU"
            print(f"[Warning] GPU not available, using CPU.")
   
        self.mode = mode
        self.projection = projection

        self.trunk = get_sei_trunk(self.quant, self.compile)
        self.head = get_sei_head_llra(k, self.quant, self.compile)
        if self.projection:
            self.proj = get_sei_projection(self.quant, mode, self.compile)

            
        self.to(self.device)

    def set_mode(self, mode):
        if mode != "sequence" and mode != "variant":
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {mode}")
        else:
            if self.projection:
                self.proj.set_mode(mode)
            self.mode = mode
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.device = device
        return self
    def forward(self, x):
        """
        Forward pass: computes output for both original and reversed input
        and averages the results. This is fed into the projector.
        """
        if self.projection:
            if self.proj.mode == "variant":
                x_r, x_a = x
                for_x_r = self.trunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                rev_x_r = self.trunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                for_x_a = self.trunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                rev_x_a = self.trunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2

                out = (out_r, out_a)
                out = self.proj(out)
            else: ## default to sequence
                for_x = self.trunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                rev_x = self.trunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2
                out = self.proj(out)
        else:
            if self.mode == "variant":
                x_r, x_a = x
                for_x_r = self.trunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                rev_x_r = self.trunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                for_x_a = self.trunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                rev_x_a = self.trunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2

                out = (out_r, out_a)
            else:
                for_x = self.trunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                rev_x = self.trunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2

        return out



import torch
import torch.nn as nn
import bitsandbytes as bnb


def convert_to_int8(model, device='cuda'):
    """
    Convert model Linear layers to INT8.
    """
    model = model.cpu().eval()  # Add eval()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module
            parts = name.rsplit('.', 1)
            parent = model if len(parts) == 1 else model.get_submodule(parts[0])
            child_name = parts[-1]
            
            # Create INT8 layer
            int8_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0
            )
            
            # Detach weights to remove gradient tracking
            int8_layer.weight = bnb.nn.Int8Params(
                module.weight.data.detach().contiguous(),
                requires_grad=False
            )
            if module.bias is not None:
                int8_layer.bias = nn.Parameter(module.bias.data.detach().clone(), requires_grad=False)
            
            setattr(parent, child_name, int8_layer)
    
    return model.to(device)
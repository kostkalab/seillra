from .get_models import get_sei_trunk_q, get_sei_head_llra, get_sei_head_llra_q
import torch.nn as nn
import seimodel as sm
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
    def __init__(self, k: int, projection: bool = True, mode: Literal["sequence", "variant"] = "sequence", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.mode = mode
        self.projection = projection
        if self.device == "cpu":
            self.trunk = get_sei_trunk_q()
            self.head = get_sei_head_llra_q(k)
        else:
            self.trunk = sm.get_sei_trunk().load_weights()
            self.head = get_sei_head_llra(k)
            
        if self.projection:
            self.proj = sm.get_sei_projection().load_weights()
            self.proj.set_mode(mode)
        self.device = device

    def set_mode(self, mode):
        if mode != "sequence" and mode != "variant":
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {mode}")
        else:
            if self.projection:
                self.proj.set_mode(mode)
            self.mode = mode
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


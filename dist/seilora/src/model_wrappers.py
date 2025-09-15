from .get_models import get_sei_trunk_q, get_sei_head_lora
import torch.nn as nn
import seimodel as sm
import torch
# import os, sys
# print(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../seimodel-dev/dist/seimodel/src')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../seimodel-dev/dist/seimodel/src')))
# import sei_projection as sm

import os, sys

# here = os.path.dirname(os.path.abspath(__file__))
# seimodel_root = os.path.abspath(os.path.join(here, '../../../../seimodel-dev/dist'))
# sys.path.append(seimodel_root)

# from seimodel.src import get_seimodels as sm
class SeiLoraWrapper(nn.Module):
    def __init__(self, k: int, projection = True, mode = "sequence"):
        super().__init__()

        self.mode = mode
        self.projection = projection
        self.qtrunk = get_sei_trunk_q()
        self.head = get_sei_head_lora(k)
        if self.projection:
            self.proj = sm.get_sei_projection().load_weights()
            self.proj.set_mode(mode)

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
            if self.projection.mode == "variant":
                x_r, x_a = x
                with torch.no_grad():
                    for_x_r = self.qtrunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                with torch.no_grad():
                    rev_x_r = self.qtrunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                with torch.no_grad():
                    for_x_a = self.qtrunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                with torch.no_grad():
                    rev_x_a = self.qtrunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2
                out = (out_r, out_a)
                out = self.proj(out)
            else: ## default to sequence
                with torch.no_grad():
                    for_x = self.qtrunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                with torch.no_grad():
                    rev_x = self.qtrunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2
                out = self.proj(out)
        else:
            if self.mode == "variant":
                x_r, x_a = x
                with torch.no_grad():
                    for_x_r = self.qtrunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                with torch.no_grad():
                    rev_x_r = self.qtrunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                with torch.no_grad():
                    for_x_a = self.qtrunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                with torch.no_grad():
                    rev_x_a = self.qtrunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2
                out = (out_r, out_a)
            else:
                with torch.no_grad():
                    for_x = self.qtrunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                with torch.no_grad():
                    rev_x = self.qtrunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2

        return out


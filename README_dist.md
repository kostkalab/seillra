
## Introduction

This package contains a low-rank approximation of (the head of) the Sei model.


## Usage

Install with

```bash
$ pip install git+https://github.com/kostkalab/seilora.git
```

and then

```python
import torch
import collections
import seilora as sl
imoprt seimodel as sm

# Get sub-models (e.g., quantized turnk, approximate head, full projection)
mod_trunk = sl.get_sei_trunk_q()
mod_head  = sl.get_sei_head_lora(k=16)
mod_projection = sm.get_sei_projection().load_weights()

#- Make a full model
mod = torch.nn.Sequential(collections.OrderedDict([
    ('trunk', mod_trunk),
    ('head', mod_head),
    ('projection', mod_projection)
]))

#...
```

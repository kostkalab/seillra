import seimodel as sm
import torch
import torch.ao.quantization.quantize_fx as quant_fx
from torch.ao.quantization import get_default_qconfig, QConfigMapping
import yaml
from importlib import resources
from  .model_loading import load_model_state_dict

CONFIG_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("config.yaml")

# - read in configuration
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)

APP_NAME = CONFIG["app_name"]
VERSION = str(CONFIG["version"])

#- model factory for quantized trunk model
def get_sei_trunk_q():
    """
    Returns a quantized SEI trunk model with weights loaded from config URLs.
    """
    stm = sm.SeiTrunk()
    stm.to('cpu')
    example_input = torch.zeros(1, 4, 4096)
    _ = stm(example_input) #- to initialize bsplines
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    prepared = quant_fx.prepare_fx(stm, qconfig_mapping, example_input)
    quantized = quant_fx.convert_fx(prepared)
    return load_model_state_dict(
        quantized,
        url_wts=CONFIG["fn_trunk_q-random-5k_wts"],
        url_wts_sha=CONFIG["fn_trunk_q-random-5k_sha"],
        app_name=APP_NAME,
        version=VERSION
    )

def get_sei_head_lora(k:int=16):
    #- a sei head lora model with rank k
    from .sei_head_lora import SeiHeadLora
    mod = SeiHeadLora(k=k)
    
    return load_model_state_dict(mod,
                                    url_wts=CONFIG[f"fn_head_lora_{k}_wts"],
                                    url_wts_sha=CONFIG[f"fn_head_lora_{k}_sha"],
                                    app_name=APP_NAME,
                                    version=VERSION)


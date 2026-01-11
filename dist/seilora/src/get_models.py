import seimodel as sm
import torch
import torch.ao.quantization.quantize_fx as quant_fx
from torch.ao.quantization import get_default_qconfig, QConfigMapping
import yaml
from importlib import resources
from  .model_loading import load_model_state_dict
import warnings

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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Please use quant_min and quant_max to specify the range for observers.*"
        )
        warnings.filterwarnings(
            "ignore",
            message="must run observer before calling calculate_qparams.*"
        )
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*"
        )
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

def get_sei_head_llra_q(k:int=16, debug = False):
    """
    Returns a quantized SEI lora head model with weights loaded from config URLs.
    """
    from .sei_head_llra import SeiHeadLLRA
    stm = SeiHeadLLRA(k=k)
    stm.to('cpu')
    example_input = torch.zeros(1, 15360)
    # _ = stm(example_input) #- to initialize bsplines
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    prepared = quant_fx.prepare_fx(stm, qconfig_mapping, example_input)
    quantized = quant_fx.convert_fx(prepared)
    label = str(k)
    if debug == True:
        return quantized
    return load_model_state_dict(
        quantized,
        url_wts=CONFIG[f"fn_head_lora_{label}_q-random-5k_wts"],
        url_wts_sha=CONFIG[f"fn_head_lora_{label}_q-random-5k_sha"],
        app_name=APP_NAME,
        version=VERSION
    )

def get_sei_head_llra(k:int=16):
    #- a sei head lora model with rank k
    from .sei_head_llra import SeiHeadLLRA
    mod = SeiHeadLLRA(k=k)
    
    label = str(k)
    
    return load_model_state_dict(mod,
                                    url_wts=CONFIG[f"fn_head_lora_{label}_wts"],
                                    url_wts_sha=CONFIG[f"fn_head_lora_{label}_sha"],
                                    app_name=APP_NAME,
                                    version=VERSION)


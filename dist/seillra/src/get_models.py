import seimodel as sm
import torch
import torch.ao.quantization.quantize_fx as quant_fx
from torch.ao.quantization import get_default_qconfig, QConfigMapping
import yaml
from importlib import resources
from  .model_loading import load_model_state_dict
import warnings
from typing import Optional, Literal
import torch.nn as nn
import bitsandbytes as bnb

CONFIG_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("config.yaml")

# - read in configuration
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)

APP_NAME = CONFIG["app_name"]
VERSION = str(CONFIG["version"])

#- model factory for quantized trunk model

    
def get_sei_trunk(quant: Literal["CPU", "GPU_fp16", "GPU_int8", None] = "CPU", compile : bool = True):
    """
    Returns a quantized SEI trunk model with weights loaded from config URLs.
    """
    if quant == "CPU":
        from .sei_parts import SeiTrunk
        stm = SeiTrunk()
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
            warnings.filterwarnings(
                "ignore",
                message="QConfig must specify a FixedQParamsObserver.*"
            )
            qconfig = get_default_qconfig("fbgemm")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            prepared = quant_fx.prepare_fx(stm, qconfig_mapping, example_input)
            quantized = quant_fx.convert_fx(prepared)
            return load_model_state_dict(
                quantized,
                url_wts=CONFIG[f"fn_trunk_q-random-5k_wts"],
                url_wts_sha=CONFIG[f"fn_trunk_q-random-5k_sha"],
                app_name=APP_NAME,
                version=VERSION
            )
    else:
        from .sei_parts import SeiTrunk
        stm = SeiTrunk()
        model = load_model_state_dict(
            stm,
            url_wts=CONFIG[f"fn_trunk_wts"],
            url_wts_sha=CONFIG[f"fn_trunk_sha"],
            app_name=APP_NAME,
            version=VERSION
        )
        if quant == "GPU_fp16":
            model= model.half()
        elif quant == "GPU_int8":
            model = convert_to_int8(model.half())
        if compile:
            model = torch.compile(model, mode="reduce-overhead")
        return model


def get_sei_head_llra(k:int|None =256, quant: Literal["CPU", "GPU_fp16", "GPU_int8", None] = "CPU", compile : bool = True):
    #- a sei head lora model with rank k
    if k is not None:
        if quant == "CPU":
            from .sei_parts import SeiHeadLLRA, QuantizedSeiHead

            stm = SeiHeadLLRA(k=k)
            stm.to('cpu')
            example_input = torch.zeros(1, 15360)
            # _ = stm(example_input) #- to initialize bsplines
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
                warnings.filterwarnings(
                    "ignore",
                    message="QConfig must specify a FixedQParamsObserver.*"
                )
                qconfig = get_default_qconfig("fbgemm")
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                prepared = quant_fx.prepare_fx(stm, qconfig_mapping, example_input)
                quantized = quant_fx.convert_fx(prepared)
                label = str(k)
                model =  load_model_state_dict(
                    quantized,
                    url_wts=CONFIG[f"fn_head_lora_{label}_q-random-5k_wts"],
                    url_wts_sha=CONFIG[f"fn_head_lora_{label}_q-random-5k_sha"],
                    app_name=APP_NAME,
                    version=VERSION
                )
                return QuantizedSeiHead(model)
        else:
            from .sei_parts import SeiHeadLLRA
            mod = SeiHeadLLRA(k=k)
            
            label = str(k)
            
            model = load_model_state_dict(mod,
                                            url_wts=CONFIG[f"fn_head_lora_{label}_wts"],
                                            url_wts_sha=CONFIG[f"fn_head_lora_{label}_sha"],
                                            app_name=APP_NAME,
                                            version=VERSION)
            if quant == "GPU_fp16":
                model= model.half()
            elif quant == "GPU_int8":
                model = convert_to_int8(model.half())
            if compile:
                model = torch.compile(model, mode="reduce-overhead")
            return model
    
    else:
        if quant == "CPU":
            from .sei_parts import SeiHead, QuantizedSeiHead

            stm = SeiHead()
            stm.to('cpu')
            example_input = torch.zeros(1, 15360)
            # _ = stm(example_input) #- to initialize bsplines
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
                warnings.filterwarnings(
                    "ignore",
                    message="QConfig must specify a FixedQParamsObserver.*"
                )
                qconfig = get_default_qconfig("fbgemm")
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                prepared = quant_fx.prepare_fx(stm, qconfig_mapping, example_input)
                quantized = quant_fx.convert_fx(prepared)
                label = str(k)
                model =  load_model_state_dict(
                    quantized,
                    url_wts=CONFIG[f"fn_head_q-random-5k_wts"],
                    url_wts_sha=CONFIG[f"fn_head_q-random-5k_sha"],
                    app_name=APP_NAME,
                    version=VERSION
                )
                return QuantizedSeiHead(model)
        else:
            from .sei_parts import SeiHead
            mod = SeiHead()
            
            model =  load_model_state_dict(mod,
                                            url_wts=CONFIG[f"fn_head_wts"],
                                            url_wts_sha=CONFIG[f"fn_head_sha"],
                                            app_name=APP_NAME,
                                            version=VERSION)
            if quant == "GPU_fp16":
                model= model.half()
            elif quant == "GPU_int8":
                model = convert_to_int8(model.half())
            if compile:
                model = torch.compile(model, mode="reduce-overhead")
            return model




def get_sei_projection(quant: Literal["CPU", "GPU_fp16", "GPU_int8", None] = "CPU", mode: Literal["sequence", "variant"] = "sequence", compile : bool = True):
    """
    Returns a quantized SEI projection model with weights loaded from config URLs.
    """
    if quant == "CPU":
        from .sei_parts import SeiProjectionQuantizable, QuantizedSeiProjection
        stm = SeiProjectionQuantizable()
        stm.to('cpu')
        example_input = torch.zeros(1, 21907)
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
            warnings.filterwarnings(
                "ignore",
                message="QConfig must specify a FixedQParamsObserver.*"
            )
            qconfig = get_default_qconfig("fbgemm")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            prepared = quant_fx.prepare_fx(stm, qconfig_mapping, example_input)
            quantized = quant_fx.convert_fx(prepared)
            model =  load_model_state_dict(
                quantized,
                url_wts=CONFIG[f"fn_projection_q-random-5k_wts"],
                url_wts_sha=CONFIG[f"fn_projection_q-random-5k_sha"],
                app_name=APP_NAME,
                version=VERSION
            )
            quantized_model = QuantizedSeiProjection(model)
            quantized_model.set_mode(mode)
            return quantized_model
    else:
        from .sei_parts import SeiProjection
        stm = SeiProjection()
        model =  load_model_state_dict(
            stm,
            url_wts=CONFIG[f"fn_projection_wts"],
            url_wts_sha=CONFIG[f"fn_projection_sha"],
            app_name=APP_NAME,
            version=VERSION
        )
        if quant == "GPU_fp16":
            model= model.half()
        elif quant == "GPU_int8":
            model = convert_to_int8(model.half())
        model.set_mode(mode)
        if compile:
            model = torch.compile(model, mode="reduce-overhead")
        return model




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

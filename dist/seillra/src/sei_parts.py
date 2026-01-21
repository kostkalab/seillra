"""
Sei architecture: Head
"""
import numpy as np
import torch
import torch.nn as nn

import numpy as np
from scipy.interpolate import splev
from importlib import resources
from importlib.abc import Traversable
import re


TARGET_ANNOT_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("target.names")

def read_target_annot(filepath: Traversable) -> dict[str, list[str]]:
    target_annot = {"context": [], "assay": [], "info": []}
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(" | ")
                if len(parts) in [3, 4]:  # - some lines have 4 fields, probably to make it uniq.
                    target_annot["context"].append(parts[0])
                    target_annot["assay"].append(parts[1])
                    target_annot["info"].append(parts[2])
                else:
                    raise ValueError(f"Malformed line in '{TARGET_ANNOT_FILE}' at line {i + 1}")
    except FileNotFoundError:
        print(f"Error: The file '{TARGET_ANNOT_FILE}' was not found.")
    return target_annot

class SeiHeadLLRA(torch.nn.Module):
    target_annot = read_target_annot(TARGET_ANNOT_FILE)
    def __init__(self, k:int=16):
        super(SeiHeadLLRA, self).__init__()

        _seiNDIM = 960*16  # number of input features after the spline layer
        _seiNPROFILES = 21907  # number of output features (number of profiles)
        self.loraw11 = torch.nn.Linear(_seiNDIM, k, bias=False)
        self.loraw12 = torch.nn.Linear(k, _seiNPROFILES)
        self.loraw21 = torch.nn.Linear(_seiNPROFILES, k, bias=False)
        self.loraw22 = torch.nn.Linear(k, _seiNPROFILES)
        self._seiNDIM = _seiNDIM
        self._seiNPROFILES = _seiNPROFILES
        self.k = k

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        # x is [B, _seiNDIM] after the spline layer
        x = self.loraw11(x)
        x = self.loraw12(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.loraw21(x)
        x = self.loraw22(x)
        x = torch.nn.functional.sigmoid(x)
        return x

    def init_weights(self, w11:torch.Tensor|None=None, w12:torch.Tensor|None=None, b1:torch.Tensor|None=None,
                     w21:torch.Tensor|None=None, w22:torch.Tensor|None=None, b2:torch.Tensor|None=None):
        """
        Initialize the weights of the linear layers in the SeiHeadLora model.
        """

        if w11 is not None:
            assert w11.shape == (self.k, self._seiNDIM), f'Expected w11 shape {(self.k, self._seiNDIM)}, got {w11.shape}'
            self.loraw11.weight.data.copy_(w11)
        if w12 is not None:
            assert w12.shape == (self._seiNPROFILES, self.k ), f'Expected w12 shape {(self._seiNPROFILES, self.k)}, got {w12.shape}'
            self.loraw12.weight.data.copy_(w12)
        if b1 is not None:
            assert b1.shape == (self._seiNPROFILES,), f'Expected b1 shape {(self._seiNPROFILES,)}, got {b1.shape}'
            self.loraw12.bias.data.copy_(b1)
        if w21 is not None:
            assert w21.shape == (self.k, self._seiNPROFILES), f'Expected w21 shape {(self.k, self._seiNPROFILES)}, got {w21.shape}'
            self.loraw21.weight.data.copy_(w21)
        if w22 is not None:
            assert w22.shape == (self._seiNPROFILES, self.k), f'Expected w22 shape {(self._seiNPROFILES, self.k)}, got {w22.shape}'
            self.loraw22.weight.data.copy_(w22)
        if b2 is not None:
            assert b2.shape == (self._seiNPROFILES,), f'Expected b2 shape {(self._seiNPROFILES,)}, got {b2.shape}'
            self.loraw22.bias.data.copy_(b2)

    def search_target_annot(
        self, pattern: str, field: str = "context", return_annot: bool = False
    ) -> list[int] | tuple[list[int], dict[str, list[str]]]:
        """
        Search for a regex pattern in the specified field of target_annot.
        Returns a list of indices where the pattern matches.
        If return_annot is True, also returns a dict with annotations for
        the matching entries.
        """

        regex = re.compile(pattern)
        # Assume target_annot is a list of dicts with keys: 'context', 'assay', 'info'
        matches = []
        for idx, annot in enumerate(self.target_annot):
            # - fixme: seems horribly inefficient to seperately search each field...
            if field in annot and regex.search(str(annot[field])):
                matches.append(idx)
        if return_annot:
            # Slice all lists in the dict to only matching indices
            matched = {k: [v[i] for i in matches] for k, v in self.target_annot.items()}
            return matches, matched
        return matches



class SeiHead(nn.Module):
    target_annot = read_target_annot(TARGET_ANNOT_FILE)
    def __init__(self, dim_ipt=15360, n_genomic_features=21907):
        super(SeiHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(dim_ipt, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        predict = self.classifier(x)
        return predict

    def search_target_annot(
        self, pattern: str, field: str = "context", return_annot: bool = False
    ) -> list[int] | tuple[list[int], dict[str, list[str]]]:
        """
        Search for a regex pattern in the specified field of target_annot.
        Returns a list of indices where the pattern matches.
        If return_annot is True, also returns a dict with annotations for
        the matching entries.
        """

        regex = re.compile(pattern)
        # Assume target_annot is a list of dicts with keys: 'context', 'assay', 'info'
        matches = []
        for idx, annot in enumerate(self.target_annot):
            # - fixme: seems horribly inefficient to seperately search each field...
            if field in annot and regex.search(str(annot[field])):
                matches.append(idx)
        if return_annot:
            # Slice all lists in the dict to only matching indices
            matched = {k: [v[i] for i in matches] for k, v in self.target_annot.items()}
            return matches, matched
        return matches




def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.

    """

    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * ( 2*(dist>0)-1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist),np.max(dist),n_knots+2)[1:-1]
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)).float()



class BSplineTransformation(nn.Module):

    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            self._spline_tr = self._spline_tr.to(input.device).to(input.dtype)
        
        return  torch.matmul(input, self._spline_tr)



class SeiTrunk(nn.Module):
    def __init__(self, sequence_length=4096):
        """
        Parameters
        ----------
        sequence_length : int
        """
        super(SeiTrunk, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True))

        self._spline_df = int(128/8)        
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False))


    def forward(self, x):
        """Forward propagation of a batch.
        """
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        
        spline_out = self.spline_tr(out)
        reshape_out = spline_out.flatten(start_dim=1)
        
        return reshape_out



CLASS_ANNOT_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("seqclass.names")
HISTONE_INDEX_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("histone_inds.npy")

def read_class_annot(filepath: resources.abc.Traversable) -> list[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f] 
    

def sc_hnorm_varianteffect(chromatin_profile_ref, chromatin_profile_alt, histone_inds, device):
    histone_inds = histone_inds.clone().to(device)
    chromatin_profile_ref_adjust = chromatin_profile_ref.clone()
    chromatin_profile_ref_adjust[:, histone_inds] = chromatin_profile_ref_adjust[:, histone_inds] * (
        (chromatin_profile_ref[:, histone_inds].sum(axis=1)*0.5 +
        chromatin_profile_alt[:, histone_inds].sum(axis=1)*0.5) /
        chromatin_profile_ref[:, histone_inds].sum(axis=1))[:, None]

    chromatin_profile_alt_adjust = chromatin_profile_alt.clone()
    chromatin_profile_alt_adjust[:, histone_inds] = chromatin_profile_alt_adjust[:, histone_inds] * (
        (chromatin_profile_ref[:, histone_inds].sum(axis=1)*0.5 +
        chromatin_profile_alt[:, histone_inds].sum(axis=1)*0.5) /
        chromatin_profile_alt[:, histone_inds].sum(axis=1))[:, None]
    return (chromatin_profile_ref_adjust, chromatin_profile_alt_adjust)
    

class SeiProjection(nn.Module):

    class_annot = read_class_annot(CLASS_ANNOT_FILE)
    histone_indices = torch.from_numpy(np.load(HISTONE_INDEX_FILE))

    def __init__(self, n_genomic_features=21907, n_classes = 61):
        """
        Parameters
        ----------
        n_genomic_features : int
        n_classes : int
        """
        super(SeiProjection, self).__init__()

        self.projector = nn.Linear(n_genomic_features, n_classes, bias=False)
        self.set_mode("sequence")

    def forward(self, x):
        """Forward propagation of a batch.
        """
        if self.mode == "sequence":
            out = self.projector(x) / self.projector.weight.norm(dim=1)
            return out
        elif self.mode == "variant":
            ref, alt = x
            ref_adj, alt_adj = sc_hnorm_varianteffect(ref, alt, SeiProjection.histone_indices, ref.device)
            ref_out = self.projector(ref_adj) / self.projector.weight.norm(dim=1)
            alt_out = self.projector(alt_adj) / self.projector.weight.norm(dim=1)
            return (ref_out, alt_out)
        else:
            print(f"Not sequence or variant, instead: {self.mode}")



    def set_mode(self, mode):
        if mode == "sequence":
            self.mode = mode
        elif mode == "variant":
            self.mode = mode
        else:
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {self.mode}")


class SeiProjectionQuantizable(nn.Module):

    class_annot = read_class_annot(CLASS_ANNOT_FILE)
    histone_indices = torch.from_numpy(np.load(HISTONE_INDEX_FILE))

    def __init__(self, n_genomic_features=21907, n_classes = 61):
        """
        Parameters
        ----------
        n_genomic_features : int
        n_classes : int
        """
        super(SeiProjectionQuantizable, self).__init__()

        self.projector = nn.Linear(n_genomic_features, n_classes, bias=False)
        self.set_mode("sequence")

    def forward(self, x):
        """Forward propagation of a batch.
        """
        if self.mode == "sequence":
            out = self.projector(x)
            return out
        elif self.mode == "variant":
            ref, alt = x
            ref_adj, alt_adj = sc_hnorm_varianteffect(ref, alt, SeiProjectionQuantizable.histone_indices, ref.device)
            ref_out = self.projector(ref_adj)
            alt_out = self.projector(alt_adj) 
            return (ref_out, alt_out)
        else:
            print(f"Not sequence or variant, instead: {self.mode}")



    def set_mode(self, mode):
        if mode == "sequence":
            self.mode = mode
        elif mode == "variant":
            self.mode = mode
        else:
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {self.mode}")


class QuantizedSeiHead(torch.nn.Module):
    target_annot = read_target_annot(TARGET_ANNOT_FILE)
    def __init__(self, model: nn.Module):
        super(QuantizedSeiHead, self).__init__()

        self.model = model

    def forward(self, x):
        return self.model(x)

    def search_target_annot(
        self, pattern: str, field: str = "context", return_annot: bool = False
    ) -> list[int] | tuple[list[int], dict[str, list[str]]]:
        """
        Search for a regex pattern in the specified field of target_annot.
        Returns a list of indices where the pattern matches.
        If return_annot is True, also returns a dict with annotations for
        the matching entries.
        """

        regex = re.compile(pattern)
        # Assume target_annot is a list of dicts with keys: 'context', 'assay', 'info'
        matches = []
        for idx, annot in enumerate(self.target_annot):
            # - fixme: seems horribly inefficient to seperately search each field...
            if field in annot and regex.search(str(annot[field])):
                matches.append(idx)
        if return_annot:
            # Slice all lists in the dict to only matching indices
            matched = {k: [v[i] for i in matches] for k, v in self.target_annot.items()}
            return matches, matched
        return matches


class QuantizedSeiProjection(nn.Module):
    class_annot = read_class_annot(CLASS_ANNOT_FILE)
    histone_indices = torch.from_numpy(np.load(HISTONE_INDEX_FILE))
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.mode = "sequence"
    
    def forward(self, x):
        if self.mode == "sequence":
            # Quantized model does the projection
            return self.model(x)
        
        elif self.mode == "variant":
            ref, alt = x
            # Preprocessing happens outside the quantized model
            ref_adj, alt_adj = sc_hnorm_varianteffect(
                ref, alt,
                QuantizedSeiProjection.histone_indices,
                ref.device
            )
            # Quantized model does the projection for each
            ref_out = self.model(ref_adj)
            alt_out = self.model(alt_adj)
            return (ref_out, alt_out)
        else:
            raise ValueError(
                f"Mode options are: 'sequence' or 'variant'. Received '{self.mode}'."
            )
    
    def set_mode(self, mode):
        if mode in ("sequence", "variant"):
            self.mode = mode
        else:
            raise ValueError(
                f"Mode options are: 'sequence' or 'variant'. Received '{mode}'."
            )

import torch

class SeiHeadLLRA(torch.nn.Module):
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
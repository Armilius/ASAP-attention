import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class TimeCausalMask():
    def __init__(self, L, device="cpu", width=9):
        mask_shape = [L, L]
        kernel = int((width - 1) // 2)
        with torch.no_grad():
            # self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask = torch.ones(mask_shape, dtype=torch.int).to(device)
            if width != 1:
                self._mask = self._mask - torch.triu(self._mask, diagonal=-kernel) + torch.triu(self._mask,
                                                                                                diagonal=kernel)
            else:
                self._mask = self._mask - torch.triu(self._mask) + torch.triu(self._mask, diagonal=1)

    @property
    def mask(self):
        return self._mask.type(torch.bool)


class DiagMask():
    def __init__(self, L, device='cpu'):
        mask_shape = [L, L]
        with torch.no_grad():
            self._mask = torch.ones(mask_shape, dtype=torch.int).to(device)
            self._mask = self._mask - torch.diag_embed(torch.diag(self._mask))

    @property
    def mask(self):
        return self._mask.type(torch.bool)


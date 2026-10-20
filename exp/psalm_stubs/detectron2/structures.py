import torch
class BoxMode:
    XYXY_ABS = 0
    XYWH_ABS = 1
class Boxes:
    def __init__(self, tensor):
        self.tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    def to(self, device):
        self.tensor = self.tensor.to(device); return self
    def __len__(self): return self.tensor.shape[0]
    def __getitem__(self, i): return Boxes(self.tensor[i])
class BitMasks:
    def __init__(self, tensor):
        self.tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    def to(self, device):
        self.tensor = self.tensor.to(device); return self
    def __len__(self): return self.tensor.shape[0]
    def __getitem__(self, i): return BitMasks(self.tensor[i])
class Instances:
    def __init__(self, image_size, **kwargs):
        object.__setattr__(self, "_image_size", image_size)
        for k, v in kwargs.items(): object.__setattr__(self, k, v)
    def __len__(self):
        for k in ("pred_masks", "gt_masks", "pred_boxes", "gt_boxes", "scores"):
            if hasattr(self, k): return len(getattr(self, k))
        return 0
    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if hasattr(v, "to"): object.__setattr__(self, k, v.to(device))
        return self
class ImageList:
    def __init__(self, tensors, image_sizes=None):
        self.tensor = torch.stack(tensors) if isinstance(tensors, (list, tuple)) else tensors
        self.image_sizes = image_sizes or [tuple(t.shape[-2:]) for t in (tensors if isinstance(tensors, (list, tuple)) else [tensors])]
    @staticmethod
    def from_tensors(tensors, size_divisibility=0, pad_value=0):
        return ImageList(tensors)
    def to(self, device):
        self.tensor = self.tensor.to(device); return self

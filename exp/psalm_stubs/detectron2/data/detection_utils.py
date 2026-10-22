"""Minimal image I/O shims for PSALM inference."""
import numpy as np
from PIL import Image

def read_image(file_name, format=None):
    img = Image.open(file_name).convert("RGB")
    return np.asarray(img)[:, :, ::-1]  # detectron2 convention: BGR

def check_image_size(dataset_dict, image):
    return None

def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    return annotation


class _Boxes:
    def __init__(self, tensor):
        self.tensor = tensor

class _Masks:
    def __init__(self, tensor):
        self.tensor = tensor
    def get_bounding_boxes(self):
        import torch
        n, h, w = self.tensor.shape
        boxes = torch.zeros((n, 4), dtype=torch.float32)
        for i in range(n):
            ys, xs = torch.where(self.tensor[i] > 0)
            if len(xs):
                boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
        return _Boxes(boxes)
    def __getitem__(self, i):
        return self.tensor[i]
    @property
    def shape(self):
        return self.tensor.shape

class _Instances:
    def __init__(self, image_size):
        self.image_size = image_size
        self.gt_masks = _Masks(__import__("torch").zeros((0, image_size[0], image_size[1])))
        self.gt_boxes = _Boxes(__import__("torch").zeros((0, 4)))
        self.gt_classes = __import__("torch").zeros((0,), dtype=__import__("torch").int64)

def _poly_to_mask(seg, h, w):
    import numpy as np, cv2
    m = np.zeros((h, w), dtype=np.float32)
    if isinstance(seg, dict):
        try:
            from pycocotools import mask as coco_mask
            m = coco_mask.decode(seg).astype(np.float32)
            if m.ndim == 3:
                m = m[:, :, 0]
            return m
        except Exception:
            return m
    polys = seg
    if isinstance(seg, list) and seg and not isinstance(seg[0], (list, tuple)):
        polys = [seg]
    for poly in polys if isinstance(polys, list) else []:
        try:
            arr = np.asarray(poly, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size < 6 or arr.size % 2:
            continue
        pts = arr.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(m, [pts], 1.0)
    return m

def annotations_to_instances(annos, image_size, mask_format="polygon"):
    import torch, numpy as np
    h, w = image_size
    masks = []
    boxes = []
    cls = []
    for ann in annos:
        seg = ann.get("segmentation", [])
        if isinstance(seg, (list, dict)) and (isinstance(seg, dict) or len(seg)):
            m = _poly_to_mask(seg, h, w)
        else:
            m = np.zeros((h, w), dtype=np.float32)
        masks.append(m)
        ys, xs = np.where(m > 0)
        if len(xs):
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        else:
            boxes.append([0, 0, 0, 0])
        cls.append(0)
    inst = _Instances(image_size)
    inst.gt_masks = _Masks(torch.tensor(np.stack(masks) if masks else np.zeros((0, h, w)), dtype=torch.float32))
    inst.gt_boxes = _Boxes(torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)))
    inst.gt_classes = torch.tensor(cls, dtype=torch.int64)
    return inst

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    return instances


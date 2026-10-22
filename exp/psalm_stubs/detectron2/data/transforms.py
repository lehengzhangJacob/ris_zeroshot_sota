"""Minimal no-op detectron2 transform shims for PSALM inference and data prep."""
class _NoOpTransform:
    def apply_image(self, img):
        return img
    def apply_segmentation(self, seg):
        return seg
    def apply_coords(self, coords):
        return coords

class ResizeShortestEdge(_NoOpTransform):
    def __init__(self, short_edge_length, max_size=None, sample_style="choice", interp=None):
        self.short_edge_length = short_edge_length
        self.max_size = max_size

class FixedSizeCrop(_NoOpTransform):
    def __init__(self, crop_size, seg_pad_value=0):
        self.crop_size = crop_size

class RandomFlip(_NoOpTransform):
    def __init__(self, *args, **kwargs):
        pass

class ResizeScale(_NoOpTransform):
    def __init__(self, *args, **kwargs):
        pass

class _Transforms:
    def apply_segmentation(self, seg):
        return seg

def apply_transform_gens(transform_gens, img):
    return img, _Transforms()

class AugmentationList(list):
    def __init__(self, augmentations):
        super().__init__(augmentations)

class AugInput:
    def __init__(self, image=None, *args, **kwargs):
        self.image = image

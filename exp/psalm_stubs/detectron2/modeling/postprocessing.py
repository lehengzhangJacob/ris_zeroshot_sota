import torch.nn.functional as F
def sem_seg_postprocess(result, img_size, output_height, output_width):
    x = result
    if x.dim() == 2: x = x[None]
    if x.dim() == 3: x = x[None]
    x = F.interpolate(x.float(), size=(output_height, output_width), mode="bilinear", align_corners=False)
    return x[0]

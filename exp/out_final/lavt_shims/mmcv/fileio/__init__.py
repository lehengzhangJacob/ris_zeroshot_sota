import torch

class FileClient:
    def __init__(self, backend="disk", **kwargs):
        self.backend = backend
    def get(self, filename):
        return filename

def load(filename, map_location="cpu"):
    return torch.load(filename, map_location=map_location)

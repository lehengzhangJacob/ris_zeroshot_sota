import os

def mkdir_or_exist(path):
    if path:
        os.makedirs(path, exist_ok=True)

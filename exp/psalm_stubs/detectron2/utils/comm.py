def get_world_size(): return 1
def get_rank(): return 0
def is_main_process(): return True
def synchronize(): return None
def all_gather(data): return [data]

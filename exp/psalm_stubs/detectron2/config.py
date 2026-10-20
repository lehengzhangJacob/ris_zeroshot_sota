def configurable(*args, **kwargs):
    def deco(fn):
        return fn
    if args and callable(args[0]):
        return args[0]
    return deco

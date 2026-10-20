class _T:
    def __getattr__(self, name):
        return type(name, (), {})
T = _T()

class _Meta:
    def set(self, **kwargs):
        return self
class MetadataCatalog:
    @staticmethod
    def get(name):
        return _Meta()
class DatasetCatalog:
    @staticmethod
    def register(name, fn):
        return None

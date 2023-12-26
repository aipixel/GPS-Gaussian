class AutoInit:
    def init(self):
        if not hasattr(self, '_AutoInit_had_init'):
            self._init()
            self._AutoInit_had_init = True

    def _init(self):
        raise NotImplementedError

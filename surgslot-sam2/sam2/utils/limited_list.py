class QueueList(list):
    """List-backed bounded queue used by SurgSLOT memory banks."""

    def __init__(self, maxlen):
        super().__init__()
        self.maxlen = maxlen

    def append(self, item):
        if len(self) >= self.maxlen:
            self.pop(0)
        super().append(item)

    def appendleft(self, item):
        if len(self) >= self.maxlen:
            self.pop(-1)
        super().insert(0, item)

    def pop(self, index=-1):
        if len(self) == 0:
            return
        return super().pop(index)

    def clear(self):
        self.__init__(self.maxlen)

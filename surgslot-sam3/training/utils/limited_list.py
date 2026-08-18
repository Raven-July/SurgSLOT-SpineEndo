class QueueList(list):
    def __init__(self, maxlen):
        super().__init__()
        self.maxlen = maxlen

    def append(self, item):
        if len(self) >= self.maxlen:
            self.pop(0)  # Remove the oldest item.
        super().append(item)  # Insert at the end.

    def appendleft(self, item):
        if len(self) >= self.maxlen:
            self.pop(-1)  # Remove the last item.
        super().insert(0, item)  # Insert at the front.

    def pop(self, index=-1):
        if len(self) == 0:
            return
        return super().pop(index)

    # Reset the list while preserving its capacity.
    def clear(self):
        self.__init__(self.maxlen)


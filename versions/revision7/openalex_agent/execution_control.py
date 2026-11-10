"""Cooperative execution stops, independent of retrieval decisions."""


class ExperimentStopped(KeyboardInterrupt):
    def __init__(self, reason="cancelled"):
        self.reason = reason
        super().__init__(reason)

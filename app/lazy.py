"""Defer building a client until first use.

The Google clients read credentials when constructed. Building them lazily
means a misconfigured cloud setting fails the request that needs it, with a
clear 503, instead of stopping the whole service from starting.
"""

import threading


class Lazy:
    def __init__(self, factory):
        self._factory = factory
        self._target = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if self._target is None:
            with self._lock:
                if self._target is None:
                    self._target = self._factory()
        return getattr(self._target, name)

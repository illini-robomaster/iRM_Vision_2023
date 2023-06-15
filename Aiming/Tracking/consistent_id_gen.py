"""Module for generating consistent id for each object."""

import threading


class ConsistentIdGenerator:
    """Return a consistent id for each object."""

    def __init__(self, lock=False):
        """Initialize the ID generator.

        Args:
            lock (bool, optional): Whether this function runs in multi-threading.
                Defaults to False for efficiency.
        """
        self.lock_flag = lock
        if self.lock_flag:
            self._lock = threading.Lock()
        self._id = 0

    def get_id(self):
        """Get a consistent int id.

        Returns:
            int: a unique id
        """
        if self.lock_flag:
            with self._lock:
                self._id += 1
                return self._id
        else:
            self._id += 1
            return self._id

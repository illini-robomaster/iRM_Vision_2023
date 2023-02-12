import threading

class ConsistentIdGenerator(object):
    def __init__(self, lock=False):
        self.lock_flag = lock
        if self.lock_flag:
            self._lock = threading.Lock()
        self._id = 0

    def get_id(self):
        if self.lock_flag:
            with self._lock:
                self._id += 1
                return self._id
        else:
            self._id += 1
            return self._id

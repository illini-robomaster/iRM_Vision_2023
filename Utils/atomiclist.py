from threading import Lock
# (fake) atomic list
# XXX: Only .copy() returns another AtomicList
class AtomicList:
    def __init__(self, *args):
        self._lst = [*args]
        self._lck = Lock()

    def append(self, value):
        with self._lck:
            return self._lst.append(value)

    def clear(self):
        with self._lck:
            return self._lst.clear()

    def copy(self):
        with self._lck:
            return AtomicList(self._lst.copy())

    def count(self, value):
        with self._lck:
            return self._lst.count(value)

    def extend(self, iterable):
        with self._lck:
            return self._lst.extend(iterable)

    def index(self, value, start=0, stop=9223372036854775807):
        with self._lck:
            return self._lst.index(value, start=start, stop=stop)

    def insert(self, ind, value):
        with self._lck:
            return self._lst.insert(ind, value)

    def pop(self, ind):
        with self._lck:
            return self._lst.pop(ind)

    def remove(self, value):
        with self._lck:
            return self._lst.remove(value)

    def reverse(self):
        with self._lck:
            self._lst.reverse()

    def sort(self, key=None, reverse=False):
        with self._lck:
            self._lst.sort(key=key, reverse=reverse)

    # Overloading
    def __contains__(self, key):
        with self._lck:
            return key in self._lst

    def __delitem__(self, key):
        return self._lst.__delitem__(key)

    def __add__(self, other):
        with self._lck:
            return self._lst + other

    def __radd__(self, other):
        with self._lck:
            return other + self._lst

    def __iadd__(self, other):
        with self._lck:
            self._lst += other

    def __mul__(self, other):
        with self._lck:
            return self._lst * other

    def __imul__(self, other):
        with self._lck:
            self._lst *= other

    def __eq__(self, other):
        with self._lck:
            return self._lst == other

    def __ne__(self, other):
        with self._lck:
            return self._lst != other

    def __gt__(self, other):
        with self._lck:
            return self._lst > other

    def __lt__(self, other):
        with self._lck:
            return self._lst < other

    def __ge__(self, other):
        with self._lck:
            return self._lst >= other

    def __le__(self, other):
        with self._lck:
            return self._lst <= other

    def __getitem__(self, int_slice):
        with self._lck:
            return self._lst[int_slice]

    def __iter__(self):
        with self._lck:
            return iter(self._lst)

    def __len__(self):
        with self._lck:
            return len(self._lst)

    def __repr__(self):
        with self._lck:
            return repr(self._lst)

    def __reversed__(self):
        with self._lck:
            return reversed(self._lst)

    def __setitem__(self, key, value):
        with self._lck:
            self._lst[key] = value

    def __sizeof__(self):
        with self._lck:
            return self._lst.__sizeof__()

if __name__ == '__main__':
    l = AtomicList(99, 98, 97, 1, 2, 3)
    ll = iter(l)
    print(ll)
    print(type(ll))

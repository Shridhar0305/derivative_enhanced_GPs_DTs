import numpy as np
import heapq
import warnings


class MutableHeap:
    """A mutable max-heap that allows updating the values of elements!."""

    def __init__(self, data, val_func=lambda x: x):
        """Initializes the MutableHeap as a max-heap."""
        self.val = []  # The heap (stores values as tuples: (value, id_))
        self.loc = {}  # Maps original index (id_) -> heap index (in self.val)
        self._val_func = val_func

        for i, x in enumerate(data):
            self.push(i, val_func(x))  # Use push to maintain max-heap property

    def push(self, id_: int, val: float) -> None:
        """Adds a new element to the max-heap. """
        if id_ in self.loc:
            raise ValueError(f"ID {id_} already exists in the heap.")

        self.loc[id_] = len(self.val)
        self.val.append((val, id_))
        self._swim(len(self.val) - 1)

    def pop_node(self):
        if not self.val:
            return None

        out_val, out_original_index = self.val[0] # Get top node value and id
        out_node = Node(out_original_index, out_val)
        self.update(out_original_index, -np.inf) # Use update to trigger sink

        return out_node

    def top_node(self):
        if not self.val:
            return None
        val, original_index = self.val[0]
        return Node(original_index, val)

    def top_value(self):
        if not self.val:
            return None
        val, original_index = self.val[0]
        return val

    def update(self, id_: int, val: float) -> None:
        if id_ not in self.loc:
            warnings.warn("No update: Index not in heap.")
            return
        loc = self.loc[id_]
        if loc == -1:
            return  

        if self.val[loc][0] > val: 
            self.val[loc] = (val, id_)
            self._sink(loc) # Move down in max-heap after decreasing value


    def _swim(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            # Comparison for max-heap swim: if child > parent, swap
            if self.val[i][0] > self.val[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sink(self, i: int) -> None:
        n = len(self.val)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i  # In max-heap, find largest among node and children

            if left < n and self.val[left][0] > self.val[largest][0]:
                largest = left
            if right < n and self.val[right][0] > self.val[largest][0]:
                largest = right

            if largest != i:
                self._swap(i, largest)
                i = largest
            else:
                break

    def _swap(self, i: int, j: int) -> None:
        """Swaps two elements in the heap and updates their locations."""
        self.val[i], self.val[j] = self.val[j], self.val[i]  # Swap values
        self.loc[self.val[i][1]] = i  # Update location of element now at i
        self.loc[self.val[j][1]] = j  # Update location of element now at j


class Node: 
    def __init__(self, id_, val):
        self.id = id_
        self.val = val

    def getval(self):
        return self.val

    def getid(self):
        return self.id
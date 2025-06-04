class SimpleMatrix:
    """Very small stand-in for scipy.sparse matrices used in tests."""

    def __init__(self, data):
        # Expect a 2D list
        self.data = [list(row) for row in data]
        self._shape = (len(self.data), len(self.data[0]) if self.data else 0)

    @property
    def shape(self):
        return self._shape

    def setdiag(self, diag_vals):
        for i, val in enumerate(diag_vals):
            if i < self._shape[0] and i < self._shape[1]:
                self.data[i][i] = val

    def __ne__(self, other):
        diff = 0
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self.data[i][j] != other.data[i][j]:
                    diff += 1
        return type("Diff", (), {"nnz": diff})()

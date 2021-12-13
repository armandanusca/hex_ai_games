###############################################################################################
# Union Find
###############################################################################################
# keep this line for cython directives
from copy import deepcopy 

cimport cython
from libc.stdlib cimport malloc, free


cdef (int, int) itot(int i):
    '''
    Converts an integer to a tuple

        Parameters:
                i (int): Integer to be converted

        Returns:
                (tuple): Tuple with the integer
    '''
    return (int(i // 13), i % 13)

cdef int ttoi((int, int) t):
    '''
    Converts a tuple to an integer

        Parameters:
                t (tuple): Tuple to be converted

        Returns:
                (int): Integer with the tuple
    '''
    return t[0] * 13 + t[1]



cdef class UnionFind:
    def __cinit__(self, n_points=169, init = True):
        self.n_points = n_points
        self.parent = <int *> malloc(n_points * sizeof(int))
        self.rank = <int *> malloc(n_points * sizeof(int))
        
        cdef int i
        if init:
            for i in range(n_points):
                self.parent[i] = i

        self._n_sets = n_points

    def __dealloc__(self):
        free(self.parent)
        free(self.rank)

    cdef UnionFind _copy(self):
        cdef UnionFind copy
        copy = UnionFind(self.n_points, False)
        copy._n_sets = self._n_sets

        cdef int i
        for i in range(self.n_points):
            copy.parent[i] = self.parent[i]
            copy.rank[i] = self.rank[i]
        return copy
        
    def __deepcopy__(self, memo):
        return self._copy()

    cdef int _find(self, int i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

    cpdef int find(self, int i): 
        return self._find(i)

    cpdef bint join(self, (int, int) i1, (int, int) j1):
        cdef int i 
        cdef int j

        i = ttoi(i1)
        j = ttoi(j1)

        cdef int root_i, root_j
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self._n_sets -= 1
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                return root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                return root_i
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
                return root_j
        else:
            return root_i

    cpdef bint connected(self, (int, int) i1, (int, int) j1):
        cdef int i 
        cdef int j

        i = ttoi(i1)
        j = ttoi(j1)

        return self.find(i) == self.find(j)


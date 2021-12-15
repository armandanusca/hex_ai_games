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

    def __dealloc__(self):
        free(self.parent)
        free(self.rank)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef UnionFind _copy(self):
        cdef UnionFind copy
        copy = UnionFind(self.n_points, False)

        cdef int i
        for i in range(self.n_points):
            copy.parent[i] = self.parent[i]
            copy.rank[i] = self.rank[i]
        return copy
    
    def __deepcopy__(self, memo):
        return self._copy()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int _find(self, int i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self._find(self.parent[i])
            return self.parent[i]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef bint join(self, tuple i1, tuple j1):
        cdef int i 
        cdef int j

        i = ttoi(i1)
        j = ttoi(j1)

        cdef int root_i, root_j
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                return True
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                return True
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
                return True
        else:
            return False

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef bint connected(self, tuple i1, tuple j1):
        cdef int i 
        cdef int j

        i = ttoi(i1)
        j = ttoi(j1)

        return self._find(i) == self._find(j)


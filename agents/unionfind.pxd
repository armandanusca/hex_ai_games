# keep this line for cython directives


cdef class UnionFind:
    """
    Notes:
        unionfind data structure specialized for finding hex connections.
        Implementation inspired by UAlberta CMPUT 275 2015 class notes.
    Attributes:
        parent (dict): Each group parent
        rank (dict): Each group rank
        groups (dict): Stores the groups and chain of cells
        ignored (list): The neighborhood of board edges has to be ignored
    """
    cdef:
        cdef int n_points
        cdef int * parent
        cdef int * rank
        cdef int _n_sets
        cdef ign1
        cdef ign2

    cdef int _find(self, int i)
    cpdef bint join(self, (int ,int) x, (int, int) y)
    cpdef int find(self, int x)
    cpdef bint connected(self, (int, int) x, (int, int) y)
    cdef UnionFind _copy(self)
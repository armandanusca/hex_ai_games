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
    cdef public:
        dict parent
        dict rank
        dict groups
        int ign1
        int ign2

    cpdef bint join(self, x, y)
    cpdef find(self, x)
    cpdef bint connected(self, x, y)
    cpdef void set_ignored_elements(self, ignore)
    cpdef dict get_groups(self)
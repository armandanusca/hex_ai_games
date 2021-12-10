###############################################################################################
# Union Find
###############################################################################################
from copy import deepcopy 

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
    
    #cpdef __deepcopy__(self, memo_dictionary):
    #     res = UnionFind()
    #     res.parent = self.parent.copy()
    #     res.rank = self.rank.copy()
    #     res.groups = self.groups.copy()
    #     res.ignored = self.ignored[:]
    #     return res

    def __init__(self, remaining_args = None):
        """
        Initialize parent and rank as empty dictionaries, we will
        lazily add items as necessary.
        """
        if not remaining_args:
            self.parent = {}
            self.rank = {}
            self.groups = {}
            self.ign1 = -1
            self.ign2 = -1

    cpdef bint join(self, x, y):
        """
        Merge the groups of x and y if they were not already,
        return False if they were already merged, true otherwise
        Args:
            x (tuple): game board cell
            y (tuple): game board cell
        """
        rep_x = self.find(x)
        rep_y = self.find(y)

        if rep_x == rep_y:
            return False
        if self.rank[rep_x] < self.rank[rep_y]:
            self.parent[rep_x] = rep_y

            self.groups[rep_y].extend(self.groups[rep_x])
            del self.groups[rep_x]
        elif self.rank[rep_x] > self.rank[rep_y]:
            self.parent[rep_y] = rep_x

            self.groups[rep_x].extend(self.groups[rep_y])
            del self.groups[rep_y]
        else:
            self.parent[rep_x] = rep_y
            self.rank[rep_y] += 1

            self.groups[rep_y].extend(self.groups[rep_x])
            del self.groups[rep_x]

        return True

    cpdef find(self, x):
        """
        Get the representative element associated with the set in
        which element x resides. Uses grandparent compression to compress
        the tree on each find operation so that future find operations are faster.
        Args:
            x (tuple): game board cell
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            if x == self.ign1 or x == self.ign2:
                self.groups[x] = []
            else:
                self.groups[x] = [x]

        px = self.parent[x]
        if x == px:
            return x

        gx = self.parent[px]
        if gx == px:
            return px

        self.parent[x] = gx

        return self.find(gx)

    cpdef bint connected(self, x, y):
        """
        Check if two elements are in the same group.
        Args:
            x (tuple): game board cell
            y (tuple): game board cell
        """
        return self.find(x) == self.find(y)

    cpdef void set_ignored_elements(self, ignore):
        """
        Elements in ignored, edges has to be ignored
        """
        self.ign1 = ignore[0]
        self.ign2 = ignore[1]

    cpdef dict get_groups(self):
        """
        Returns:
            Groups
        """
        return self.groups

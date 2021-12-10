class UnionFind:
    """
    UnionFind data structure used for finding hex connections by grouping disjoint sets of connected nodes.
    It is a tree-based implementation with union by rank and grandparent path compression.
    Class members:
        parent: dictionary mapping node to parent
        rank: dictionary mapping group to rank
        groups: dictionary mapping group to member nodes
        ignored_nodes: board edges which are to be ignored
    """

    def __init__(self) -> None:
        """
        Initialize parent, rank and group as empty dictionaries.
        """
        self.parent = {}
        self.rank = {}
        self.groups = {}
        self.ignored_nodes = []

    def join(self, x, y) -> bool:
        """
        Join the group where x belongs with the group where y belongs, unless x and y are in the same group.
        Union by rank is used to determine which set is joined to which.
        Arguments:
            x: board cell representation
            y: board cell representation
        Return: 
            False if groups were already join
            True otherwise
        """
        group_x = self.find(x)
        group_y = self.find(y)

        if group_x == group_y:
            return False
        if self.rank[group_x] < self.rank[group_y]:
            self.parent[group_x] = group_y
            self.groups[group_y] += self.groups[group_x]
            del self.groups[group_x]
        elif self.rank[group_x] > self.rank[group_y]:
            self.parent[group_y] = group_x
            self.groups[group_x] += self.groups[group_y]
            del self.groups[group_y]
        else:
            self.parent[group_x] = group_y
            self.rank[group_y] += 1
            self.groups[group_y] += self.groups[group_x]
            del self.groups[group_x]

        return True

    def find(self, x):
        """
        Get the representation of the set to which the node x belongs. 
        Grandparent path compression is used to compress the tree structure on 
        each find operation making subsequent find operations faster.
        Lazily adds x if it is not present in any group.
        Arguments:
            x: board cell representation
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            if x in self.ignored_nodes:
                self.groups[x] = []
            else:
                self.groups[x] = [x]

        parent_x = self.parent[x]
        if x == parent_x:
            return x

        grandparent_x = self.parent[parent_x]
        if grandparent_x == parent_x:
            return parent_x

        # Path compression
        self.parent[x] = grandparent_x

        return self.find(grandparent_x)

    def connected(self, x, y) -> bool:
        """
        Check if two nodes belong to the same group.
        Args:
            x: board cell representation
            y: board cell representation
        """
        return self.find(x) == self.find(y)

    def set_ignored_nodes(self, ignored_nodes):
        """
        Set the nodes which are to be ignored. This should be the board edges
        """
        self.ignored_nodes = ignored_nodes

    def get_groups(self) -> dict:
        """
        Return:
            all groups
        """
        return self.groups

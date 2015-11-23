__author__ = 'matt'
__eamil__ = 'mia2n4@mst.edu'

from Queue import Queue

class Tree:
    def __init__(self, data, depth):
        self.data = data
        self.children = []
        self.depth = depth

    def add_child(self, data):
        self.children.append(data)

    @staticmethod
    def find_depth(tree):
        if tree.children == []:
            return 0
        else:
            return max(Tree.find_depth(tree.children[0]), Tree.find_depth(tree.children[1])) + 1

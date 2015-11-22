__author__ = 'matt'
__eamil__ = 'mia2n4@mst.edu'


class Tree:
    def __init__(self, data, depth):
        self.data = data
        self.children = []
        self.depth = depth

    def add_child(self, data):
        self.children.append(data)

__author__ = 'matt'
__eamil__ = 'mia2n4@mst.edu'

from Queue import Queue
import random
import copy


class Tree:
    def __init__(self, data, depth):
        self.data = data
        self.children = []
        self.depth = depth

    def add_child(self, data):
        self.children.append(data)

    # Find the depth of a tree
    @staticmethod
    def find_depth(tree):
        if tree.children == []:
            return 0
        else:
            return max(Tree.find_depth(tree.children[0]), Tree.find_depth(tree.children[1])) + 1

    # Select a random node from the tree
    def rand_node(self):
        nodes = Queue()
        nodes.put(self)
        num_nodes = 1
        selected = self

        # Randomly pick an element
        while not nodes.empty():
            n = nodes.get()
            if random.randint(1, num_nodes) == num_nodes:
                selected = n
            num_nodes += 1
            for i in n.children:
                nodes.put(i)

        return selected

    @staticmethod
    def crossover(parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        sel1 = child1.rand_node()
        sel2 = child2.rand_node()

        q = Queue()
        q.put(child1)
        while not q.empty():
            node = q.get()
            for i in range(len(node.children)):
                q.put(node.children[i])
                if node.children[i] is sel1:
                    node.children[i] = sel2
                    break

        q = Queue()
        q.put(child2)
        while not q.empty():
            node = q.get()
            for i in range(len(node.children)):
                q.put(node.children[i])
                if node.children[i] is sel2:
                    node.children[i] = sel1
                    break

        return child1, child2

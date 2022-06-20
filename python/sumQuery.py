""" 2D Segment Tree for range sum query

Author : Nabil Ibtehaz
Author : Rickard Norlander
"""

import numpy as np

# PartialBoth and partialY propagate upwards to the root y-node.
# PartialX and fullBoth do not, they are set only on the last node we reach in our update.
#
# fullBoth and partialY are updated when a node is fully inside the x-part of the update
# PartialX and partialBoth are updated when a node is partially inside.
#
# PartialBoth and partialY are returned when we find a fully contained node
# Quindlewap is returned when we find a fully y-contained node.
# When a node is partially y-contained then we add fullBoth and, if x-contained also partialX,
#   and we keep going.
#
#
# Update
# ------
#               y-contained    y-partial
#
# x-contained    fullBoth      partialY
#  x-partial     partialX     partialBoth
#
#
# Query
# -----
#               y-contained    y-partial
#
# x-contained   partialBoth     partialX
#  x-partial     partialY       fullBoth
#

class Node(object):
    def __init__(self):
        self.partialBoth = 0
        self.partialY = 0
        self.partialX = 0
        self.fullBoth = 0

class SegmentTree2D(object):

    def __init__(self, n, m):
        """
        Initializes the tree

        Arguments:
            n {int} -- number of rows in the matrix
            m {int} -- number of columns in the matrix
        """
        self.n = n
        self.m = m
        self.tree = []
        for i in range(4 * n):
            row = []
            for j in range(4 * m):
                row.append(Node())
            self.tree.append(row)

    def update(self, qxLo, qxHi, qyLo, qyHi, v):
        """
        Updates the tree by adding v to all elements within [qxLo:qxHi,qyLo:qyHi]

        Arguments:
            qxLo {int} -- start of x dimension
            qxHi {int} -- end of x dimension
            qyLo {int} -- start of y dimension
            qyHi {int} -- end of y dimension
            v {float/int} -- value to be added
        """
        self.updateByX((0, 0), 0, self.n-1, qxLo, qxHi, qyLo, qyHi, v)

    def query(self, qxLo, qxHi, qyLo, qyHi):
        """
        Queries the sum of all elements within [qxLo:qxHi,qyLo:qyHi]

        Arguments:
            qxLo {int} -- start of x dimension
            qxHi {int} -- end of x dimension
            qyLo {int} -- start of y dimension
            qyHi {int} -- end of y dimension
        """
        return self.queryByX((0, 0), 0, self.n-1, qxLo, qxHi, qyLo, qyHi)

    def updateByX(self, nodeID, xLo, xHi, qxLo, qxHi, qyLo, qyHi, v):
        """
        Updates along x dimension

        Arguments:
            nodeID {tuple} -- (index along 1st layer tree, index along 2nd layer tree)
            xLo {int} -- start of x dimension of the region under the node
            xHi {int} -- end of x dimension of the region under the node
            qxLo {int} -- start of x dimension of update region
            qxHi {int} -- end of x dimension of update region
            qyLo {int} -- start of y dimension of update region
            qyHi {int} -- end of y dimension of update region
            v {float/int} -- value to be added
        """
        if qxHi < xLo or xHi < qxLo:
            # X-disjoint
            return
        if qxLo <= xLo and xHi <= qxHi:
            # Fully inside query. Done with x-dimension.
            self.updateByY(nodeID, xLo, xHi, 0, self.m-1, qxLo, qxHi, qyLo, qyHi, v, True)
            return

        # Now know that node is partially inside query.
        xMid = (xLo + xHi) // 2
        left = (nodeID[0] * 2 + 1, nodeID[1])
        right = (left[0] + 1, left[1])

        # Update children.
        self.updateByX(left, xLo, xMid, qxLo, qxHi, qyLo, qyHi, v)
        self.updateByX(right, xMid + 1, xHi, qxLo, qxHi, qyLo, qyHi, v)

        # Also update this node itself.
        self.updateByY(nodeID, xLo, xHi, 0, self.m-1, qxLo, qxHi, qyLo, qyHi, v , False)

    def updateByY(self, nodeID, xLo, xHi, yLo, yHi, qxLo, qxHi, qyLo, qyHi, v, covered):
        """
        Updates along y dimension

        Arguments:
            nodeID {tuple} -- (index along 1st layer tree, index along 2nd layer tree)
            xLo {int} -- start of x dimension of the region under the node
            xHi {int} -- end of x dimension of the region under the node
            yLo {int} -- start of y dimension of the region under the node
            yHi {int} -- end of y dimension of the region under the node
            qxLo {int} -- start of x dimension of update region
            qxHi {int} -- end of x dimension of update region
            qyLo {int} -- start of y dimension of update region
            qyHi {int} -- end of y dimension of update region
            v {float/int} -- value to be added, scaled appropriately
        """
        if qyHi < yLo or yHi < qyLo:
            # Y-disjoint
            return
        node = self.tree[nodeID[0]][nodeID[1]]
        if qyLo <= yLo and yHi <= qyHi:
            # Fully inside on y-dimension.
            if covered:
                # Fully inside on both dimensions.
                node.fullBoth += v
                node.partialY += v * (yHi - yLo + 1)
            else:
                # Fully inside on y but not x.
                txLo = max(qxLo, xLo)
                txHi = min(qxHi, xHi)
                node.partialX += v * (txHi - txLo + 1)
                node.partialBoth += v * (txHi - txLo + 1) * (yHi - yLo + 1)
        else:
            yMid = (yLo + yHi) // 2
            leftID = (nodeID[0], nodeID[1] * 2 + 1)
            rightID = (leftID[0], leftID[1] + 1)

            self.updateByY(leftID, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi, v, covered)
            self.updateByY(rightID, xLo, xHi, yMid + 1, yHi, qxLo, qxHi, qyLo, qyHi, v, covered)

            left_node = self.tree[leftID[0]][leftID[1]]
            right_node = self.tree[rightID[0]][rightID[1]]

            node.partialBoth = left_node.partialBoth + right_node.partialBoth + node.partialX * (yHi - yLo + 1)
            node.partialY = left_node.partialY + right_node.partialY + node.fullBoth * (yHi - yLo + 1)

    def queryByX(self, nodeID, xLo, xHi, qxLo, qxHi, qyLo, qyHi):
        """
        Queries along x dimension

        Arguments:
            nodeID {tuple} -- (index along 1st layer tree, index along 2nd layer tree)
            xLo {int} -- start of x dimension of the region under the node
            xHi {int} -- end of x dimension of the region under the node
            qxLo {int} -- start of x dimension of update region
            qxHi {int} -- end of x dimension of update region
            qyLo {int} -- start of y dimension of update region
            qyHi {int} -- end of y dimension of update region
        """
        if qxHi < xLo or xHi < qxLo:
            # X-disjoint
            return 0
        if qxLo <= xLo and xHi <= qxHi:
            # Fully inside query. Done with x-dimension.
            return self.queryByY(nodeID, xLo, xHi, 0, self.m-1, qxLo, qxHi, qyLo, qyHi)

        # Now know that node is partially inside query.
        xMid = (xLo + xHi) // 2
        left = (nodeID[0] * 2 + 1, nodeID[1])
        right = (left[0] + 1, left[1])

        left_result = self.queryByX(left, xLo, xMid, qxLo, qxHi, qyLo, qyHi)
        right_result = self.queryByX(right, xMid + 1, xHi, qxLo, qxHi, qyLo, qyHi)

        txLo = max(qxLo, xLo)
        txHi = min(qxHi, xHi)

        this_result = self.queryByY(nodeID, xLo, xHi, 0, self.m-1, txLo, txHi, qyLo, qyHi)
        return left_result + right_result + this_result

    def queryByY(self, nodeID, xLo, xHi, yLo, yHi, qxLo, qxHi, qyLo, qyHi):
        """
        Queries along y dimension

        Arguments:
            nodeID {tuple} -- (index along 1st layer tree, index along 2nd layer tree)
            xLo {int} -- start of x dimension of the region under the node
            xHi {int} -- end of x dimension of the region under the node
            yLo {int} -- start of y dimension of the region under the node
            yHi {int} -- end of y dimension of the region under the node
            qxLo {int} -- start of x dimension of update region
            qxHi {int} -- end of x dimension of update region
            qyLo {int} -- start of y dimension of update region
            qyHi {int} -- end of y dimension of update region
        """
        if qyHi < yLo or yHi < qyLo:
            # Y-disjoint
            return 0

        node = self.tree[nodeID[0]][nodeID[1]]
        if qyLo <= yLo and yHi <= qyHi:
            # Fully inside on y-dimension.
            if qxLo <= xLo and xHi <= qxHi:
                # Fully inside on both dimensions.
                return node.partialBoth + node.partialY * (xHi - xLo + 1)
            else:
                # Fully inside on y but not x.
                scaled_value = node.partialY * (qxHi - qxLo + 1)
                return scaled_value

        # Now know that node is partially inside query on y-dimension.
        yMid = (yLo + yHi) // 2
        left = (nodeID[0], nodeID[1] * 2 + 1)
        right = (left[0], left[1] + 1)

        tyLo = max(yLo, qyLo)
        tyHi = min(yHi, qyHi)
        txLo = max(xLo, qxLo)
        txHi = min(xHi, qxHi)
        lazy_result = node.fullBoth * (txHi - txLo + 1) * (tyHi - tyLo + 1)

        if qxLo <= xLo and xHi <= qxHi:
            lazy_result += node.partialX * (tyHi - tyLo + 1)

        left_result = self.queryByY(left, xLo, xHi, yLo, yMid, qxLo, qxHi, qyLo, qyHi)
        right_result = self.queryByY(right, xLo, xHi, yMid + 1, yHi, qxLo, qxHi, qyLo, qyHi)
        return left_result + right_result + lazy_result

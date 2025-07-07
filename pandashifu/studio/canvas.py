import numpy as np
import pandas as pd


class PSNode:

    def __init__(self, pos, ntype, before=None, levels=[], info={}):

        self.dx = 10
        self.dy = -4

        self.pos = pos
        self.ntype = ntype
        self.before = before
        self.after = []
        self.nbranch = 0
        self.offset = None

        self.levels = levels

        self.info = info

    def get_root(self):

        node = self.before
        if node is None:
            return self
        else:
            return node.get_root()

    def get_all(self):

        output = [self]
        for node in self.after:
            output.extend(node.get_all())

        return output

    def get_all_lines(self):

        x = []
        y = []
        for node in self.after:
            x1, x2 = self.pos[0], node.pos[0]
            y1, y2 = self.pos[1], node.pos[1]
            if x1 == x2:
                x.extend([np.nan, x1, x2, np.nan])
                y.extend([np.nan, y1, y2, np.nan])
            else:
                xmid = x1 + (x2 - x1)*np.concatenate([np.linspace(0, 0.6, 60), np.linspace(0.61, 1, 5)])
                x.extend([np.nan] + list(xmid) + [np.nan])
                ymid = y2 + (y1 - y2)/(1 + np.exp(4*(xmid - (0.6*x1 + 0.4*x2))))
                y.extend([np.nan] + list(ymid) + [np.nan])
            data = node.get_all_lines()
            x.extend(data[0])
            y.extend(data[1])

        return x, y

    def content_label(self):

        if self.ntype == 'data':
            name, data = self.content
            return f'<b>{name}</b>:<br>{data.shape[0]} x {data.shape[1]}'
        elif self.ntype == 'visual':
            name, _, vtype = self.content
            return f'{vtype} for <b>{name}</b>'
        elif self.ntype == 'model':
            return self.content[2]
        else:
            return 'Unknown'

    def grow(self, ntype, info):

        x = self.pos[0] + self.dx
        y0 = self.pos[1]
        levels = self.levels + [self.nbranch]
        root = self.get_root()
        all_nodes = root.get_all()
        nlevels = len(self.levels)
        upper = []
        lower = []
        lower_nodes = []
        for node in all_nodes:
            if nlevels + 1 == len(node.levels) and nlevels > 0:
                shift = np.array(self.levels) - np.array(node.levels[:nlevels])
                nonzero_indices = np.nonzero(shift)[0]
                if len(nonzero_indices) > 0:
                    if shift[nonzero_indices[0]] > 0:
                        upper.append(node.pos[1])
                    else:
                        lower_nodes.append(node)
                        lower.append(node.pos[1])
        if upper:
            y0 = min(upper) + self.dy if y0 >= min(upper) else y0
        y = y0 + self.dy*self.nbranch
        if lower:
            ymax = max(lower)
            if y <= ymax:
                offset = y - ymax + self.dy
                for node in lower_nodes:
                    node.pos = (node.pos[0], node.pos[1] + offset)

        #next_node = PSNode((x, y), ntype, self, levels, content, code=code)
        next_node = PSNode((x, y), ntype, self, levels, info=info)
        self.after.append(next_node)
        self.nbranch += 1

        return next_node
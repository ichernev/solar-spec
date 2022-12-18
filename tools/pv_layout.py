import argparse
import sys

def parse(args):
    parser = argparse.ArgumentParser(
        "Compute optimal placement for panels on given area")
    parser.add_argument("-a", "--area", type=str, action="append",
                        required=True, help="rectangular area AxB in mm,cm,m")
    parser.add_argument("-p", "--panel", type=str, action="append",
                        required=True, help="panel size in AxB in mm,cm,m")
    parser.add_argument("-b", "--buffer", type=str, default="20cm",
                        help="buffer area around roof edges to avoid")
    parser.add_argument("-s", "--spacing", type=str, default="2cm",
                        help="spacing between panels in mm,cm")

    return parser.parse_args(args)

class Size:
    def __init__(self, w, h=None):
        """w,h in mm"""
        self.w = int(w)
        self.h = int(h if h else w)

    def flip(self):
        return self.__class__(self.h, self.w)

    def portrait(self):
        if self.w <= self.h:
            return self
        return self.flip()

    def landscape(self):
        return self.portrait().flip()

#     def grow(self, amt):
#         return self.__class__(self.w + amt, self.h + amt)

#     def shrink(self, amt):
#         return self.grow(-amt)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.w + other.w, self.h + other.h)
        elif isinstance(other, int):
            return self.__class__(self.w + other, self.h + other)
        else:
            raise ValueError("can't add Size with {type(other)}")

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            return self.__class__(self.w * other[0], self.h * other[1])
        elif isinstance(other, int):
            return self.__class__(self.w * other, self.h * other)

    def fit(self, other):
        if isinstance(other, self.__class__):
            return (self.w / other.w, self.h / other.h)
        elif isinstance(other, (int, float)):
            return (self.w / other, self.h / other)

    def __neg__(self):
        return self.__class__(-self.w, -self.h)

    def __str__(self):
        return f"Size({self.w}, {self.h})"

    @staticmethod
    def raw_parse_quant(q):
        if q.endswith('mm'):
            return float(q[:-2]), 'mm'
        elif q.endswith('cm'):
            return float(q[:-2]), 'cm'
        elif q.endswith('m'):
            return float(q[:-1]), 'm'
        else:
            return float(q), None

    @staticmethod
    def scale(q, m):
        if m == 'mm':
            return int(q)
        elif m == 'cm':
            return int(q * 10)
        elif m == 'm':
            return int(q * 1000)

    @classmethod
    def parse_scale(cls, q):
        l, m = cls.raw_parse_quant(q)
        if m is None:
            m = 'mm'
        return cls.scale(l, m)

    @classmethod
    def from_str(cls, axb):
        a, b = axb.split('x')
        ax, am = cls.raw_parse_quant(a)
        bx, bm = cls.raw_parse_quant(b)

        if am is None and bm is None:
            am = bm = 'mm'
        elif am is None:
            am = bm
        elif bm is None:
            bm = am

        return cls(cls.scale(ax, am), cls.scale(bx, bm))


class Fit:
    def __init__(self, panel, area, buffer, spacing, gray=50):
        """gray area in mm"""
        self._panel = panel
        self._area = area
        self._buffer = buffer
        self._spacing = spacing
        self._gray = gray

        self._amt = None

    def fit(self):
        panel_ = self._panel + self._spacing
        area_ = self._area - self._buffer

        res = area_.fit(panel_)
        self._amt = (int(res[0]), int(res[1]))
        return self._amt

    def describe(self):
        cov = self._panel * self._amt
        spc = Size(self._spacing) * (max(self._amt[0] - 1, 0), max(self._amt[1] - 1, 0))
        left = self._area - cov - spc

        # try to fit one more
        left_x = left - self._panel - Size(self._spacing)
        # import pdb; pdb.set_trace()
        print(f"fitting {self._panel} in {self._area}")
        print(f"  width:  {self._amt[0]} --> {cov.w} {spc.w}, left {left.w / 2}/{self._buffer} [{left_x.w / 2}]")
        print(f"  height: {self._amt[1]} --> {cov.h} {spc.h}, left {left.h / 2}/{self._buffer} [{left_x.h / 2}]")


# def fit(area, panel, buffer, spacing):
#     panel_ = panel + spacing
#     area_ = area - buffer

#     res = area_.fit(panel_)
#     return int(res[0]), int(res[1])

def main(args):
    opts = parse(args)

    areas = [Size.from_str(area) for area in opts.area]
    panels = [Size.from_str(panel) for panel in opts.panel]
    buffer = Size.parse_scale(opts.buffer)
    spacing = Size.parse_scale(opts.spacing)

    for base_panel in panels:
        area_cnts = []
        for area in areas:
            best = None
            for panel in (base_panel.portrait(),
                          base_panel.landscape()):
                fit = Fit(panel, area, buffer, spacing)
                res = fit.fit()
                fit.describe()
                if best is None or (best[0] * best[1] < res[0] * res[1]):
                    best = res
            area_cnts.append(best)
        area_tot = sum(c[0] * c[1] for c in area_cnts)
        print(f"panel: {base_panel} -- {area_tot}pcs : {area_cnts}")


if __name__ == '__main__':
    main(sys.argv[1:])

from argparse import ArgumentParser
from pydantic import BaseModel, validator, Field
from typing import Optional, List
from pathlib import Path
import enum
import itertools
import sys
import json
import yaml


class CurlyPattern:
    def __init__(self, pattern):
        self.p = pattern
        self.idx = []

        self._parse()

    def _parse(self):
        active = -1
        for i, c in enumerate(self.p):
            if c == '{':
                active += 1
                self._mark(active, i)
            elif c == ',':
                self._mark(active, i)
            elif c == '}':
                self._pack(active, i)
                active -= 1

    def _mark(self, active, i):
        assert 0 <= active and active <= len(self.idx)
        if active == len(self.idx):
            self.idx.append([])
        self.idx[active].append(i)

    def _pack(self, active, i):
        self._mark(active, i)
        a = self.idx[active]
        for i in range(len(a)-1, -2, -1):
            if i < 0 or not isinstance(a[i], int):
                break
        # print(f"packing {i} {len(a)} {a} {a[i+1:]}")
        self.idx[active] = a[0:i+1] + [a[i+1:]]

    def _bsrch(self, lvl, it):
        if lvl >= len(self.idx):
            return 0

        a = self.idx[lvl]
        # print(f"bsrc {lvl} {it} {a}")

        l, r, m = -1, len(a), None
        while r - l > 1:
            m = (l + r) // 2
            if it > a[m][0]:
                l = m
            else:
                r = m
        # print(f"---> {r}")
        return r

    def expand(self):
        return self.items_(0, 0, len(self.p))

    def items_(self, level, fr, to):
        fr_i = self._bsrch(level, fr)
        to_i = self._bsrch(level, to)

        iters = []
        for i in range(fr_i, to_i):
            it = []
            for lb, ub in zip(self.idx[level][i][0:], self.idx[level][i][1:]):
                it = itertools.chain(it, self.items_(level + 1, lb+1, ub))
            iters.append(it)

        sidx = []
        slots = []
        start = fr
        for i in range(fr_i, to_i):
            if start < self.idx[level][i][0]:
                slots.append(self.p[start:self.idx[level][i][0]])
            sidx.append(len(slots))
            slots.append(None)
            start = self.idx[level][i][-1] + 1

        if start < to:
            slots.append(self.p[start:to])

        for pcs in itertools.product(*iters):
            assert len(pcs) == to_i - fr_i
            for idx, i in enumerate(range(fr_i, to_i)):
                slots[sidx[idx]] = pcs[idx]
            yield "".join(slots)


class Mapping(BaseModel):
    class Target(BaseModel):
        items: List[str] = Field(description="list of target schema paths to fill in")

        @validator('items', pre=True)
        def parse_items(cls, v):
            if isinstance(v, str):
                return CurlyPattern(v).expand()
            return v

    target: Target = Field(description="where to store extracted data")

    class Source(BaseModel):
        class Matcher(BaseModel):
            exact: Optional[str]
            includes: Optional[str]
            regex: Optional[str]
        section: Optional[Matcher]
        row: Optional[Matcher]
        col: Optional[Matcher]
        const: Optional[str]
    source: Source

    class Fns(enum.Enum):
        # Use the whole input, and assign to output
        assign = "assign"
        # Extract all integers from the string
        integers = "integers"
        # Extract all numbers (floats) from the string
        numbers = "numbers"
        # Use the groups of the provided col.regex
        regex_groups = "regex_groups"
    fn: Fns = Field(Fns.assign, description="How to process the source into the target")


class MapperSpec(BaseModel):
    mappings: List[Mapping] = Field(..., description="list of mappings to apply in order")


def parse(args):
    parser = ArgumentParser("map structured data to json schema")
    parser.add_argument('--input', action="append", help="input file (can be repeated)")
    parser.add_argument('--schema', type=str, help="Path to json schema to fill in")
    parser.add_argument('--mapper-spec-path', type=str, help="Path to mapper specification")
    # parser.add_argument('--mapper-spec-path-stem', type=str,
    #                     help="index non-root property in the mapper-spec-path")
    parser.add_argument('--mapper-spec-json', type=str, help="json-encoded mapper spec")
    parser.add_argument('--output', type=str, help="where to store output file")

    return parser.parse_args(args)


def load_csv(file):
    import csv
    res = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            res.append(dict(zip(header, row)))

    if not res:
        return None

    if 'section' in res[0]:
        # fill in inferred sections
        prev_s = ''
        for item in res:
            if not item['section']:
                item['section'] = prev_s
            else:
                prev_s = item['section']

    return res

def load_inputs(opts):
    for inp in opts.input:
        if inp.endswith('.csv'):
            res = load_csv(inp)
            # TODO: Handle multiple inputs
            return res


def deep_get(obj, keypath):
    tmp = obj
    for piece in keypath.split('.'):
        if piece.isdigit():
            tmp = tmp[int(piece)]
        else:
            tmp = tmp[piece]
    return tmp


def load_spec(opts):
    if opts.mapper_spec_json:
        raw_spec = json.loads(opts.mapper_spec_json)
    elif opts.mapper_spec_path:
        path = opts.mapper_spec_path
        stem = None
        if ':' in path:
            path, stem = path.rsplit(':', 1)
        if path.endswith('.yaml'):
            raw_spec = yaml.load(Path(path).read_bytes(), Loader=yaml.Loader)
        if stem:
            raw_spec = deep_get(raw_spec, stem)
    if 'mappings' not in raw_spec:
        raw_spec = {'mappings': raw_spec}
    return MapperSpec(**raw_spec)


def main(args):
    opts = parse(args)

    data = load_inputs(opts)
    spec = load_spec(opts)
    schema = json.loads(Path(opts.schema).read_text())
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main(sys.argv[1:])
    # cp = CurlyPattern(sys.argv[1])
    # # print(cp.idx)
    # # print(list(cp.items(0, 0, len(cp.p))))
    # print(list(cp.expand()))

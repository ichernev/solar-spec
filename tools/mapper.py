#!/usr/bin/env python3

from argparse import ArgumentParser
from pydantic import BaseModel, validator, Field, root_validator
from typing import Optional, List
from pathlib import Path
import enum
import itertools
import sys
import json
import yaml
import re
import pprint


def log(*args, **kwargs):
    if 'file' not in kwargs:
        kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def log_pp(obj):
    pprint.pprint(obj, stream=sys.stderr)


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

            @root_validator
            def one_variant(cls, values):
                if sum(int(bool(values.get(k))) for k in ['exact', 'includes', 'regex']) != 1:
                    raise ValueError("exactly one of exact,includes,regex should be present")
                return values

            def matches(self, s):
                if self.exact:
                    return self.exact == s
                elif self.includes:
                    # this is case-insensitive
                    return self.includes.lower() in s.lower()
                else:
                    return bool(re.search(self.regex, s))

            def regex_match(self, s):
                if not self.regex:
                    raise ValueError("regex is not set")
                return re.search(self.regex, s)

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
    actions: List[Mapping] = Field(..., description="list of actions to apply in order")


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
        if tmp is None:
            return None
        if piece.isdigit():
            if not isinstance(tmp, list):
                tmp = None
            else:
                tmp = tmp[int(piece)]
        else:
            tmp = tmp.get(piece)

    return tmp


def deep_set(obj, keypath, val):
    tmp = obj
    path = keypath.split('.')
    for piece in path[:-1]:
        if piece.isdigit():
            raise ValueError("array indexing not supported ATM")
        if piece not in tmp:
            tmp[piece] = {}

        tmp = tmp[piece]

    tmp[path[-1]] = val


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
    if 'actions' not in raw_spec:
        raw_spec = {'actions': raw_spec}
    return MapperSpec(**raw_spec)


def schema_deep_get(schema, path):
    # start with props of root object
    props = schema['properties']
    for piece in path.split('.'):
        if piece.isdigit():
            raise ValueError("array indexing not supported ATM")
        props = props[piece]
        if props.get('allOf'):
            path = props['allOf'][0]['$ref'][2:].replace('/', '.')
            props = deep_get(schema, path)['properties']

    if 'type' not in props:
        raise ValueError(f"expected leaf property at {path}")
    return props


def load_schema(path):
    import importlib
    pcs = path.split('.')
    pref = '.'.join(['specs'] + pcs[:-1])
    cls_name = pcs[-1]
    log(f"importing {pref}")
    # import pdb; pdb.set_trace()
    proj_root = str(Path(__file__).parent.parent)
    if proj_root not in sys.path:
        sys.path.append(proj_root)
    # import pdb; pdb.set_trace()
    mod = importlib.import_module(pref)
    cls = getattr(mod, cls_name)
    return cls.schema(), cls


def main(args):
    opts = parse(args)

    data = load_inputs(opts)
    spec = load_spec(opts)
    schema, model_cls = load_schema(opts.schema)

    pre_parsed = [{} for _ in range(len(data[0]) - 2)]

    for mapping in spec.actions:
        targ_props = [schema_deep_get(schema, item) for item in mapping.target.items]

        if mapping.source.const:
            assert len(mapping.target.items) == 1
            for pp in pre_parsed:
                deep_set(pp, mapping.target.items[0], mapping.source.const)
            continue

        rows = list(data)
        matcher_section = mapping.source.section
        if matcher_section:
            rows = filter(lambda r: matcher_section.matches(r['section']), rows)
        matcher_row = mapping.source.row
        if matcher_row:
            rows = filter(lambda r: matcher_row.matches(r['property']), rows)
        matcher_col = mapping.source.col
        if matcher_col:
            rows = filter(lambda r: all(matcher_col.matches(v) for k, v in r.items()
                                        if k.startswith('col:')), rows)
        rows = list(rows)

        if len(rows) >= 2:
            log(f"section+row matches too many rows {matcher_section} {matcher_row}, skipping")
            continue

        if len(rows) == 0:
            continue

        row = rows[0]
        # only fill in cols for which all properties are None (not filled yet)
        log(f"----- {mapping}")
        for i, pp in enumerate(pre_parsed):
            if not all(deep_get(pp, item) is None for item in mapping.target.items):
                continue
            col = row[f'col:{i}']

            Fns = Mapping.Fns
            if mapping.fn == Fns.assign:
                for item in mapping.target.items:
                    deep_set(pp, item, col)
            elif mapping.fn in (Fns.integers, Fns.numbers):
                pat = r'\d+' if mapping.fn == Fns.integers else f'\d+(?:\.\d+)?'
                matches = re.findall(pat, col)
                log(f"{mapping} {matches} {pat}")
                if len(targ_props) == 1 and targ_props[0]['type'] == 'array':
                    deep_set(pp, mapping.target.items[0], matches)
                elif len(matches) == len(mapping.target.items):
                    for val, item in zip(matches, mapping.target.items):
                        deep_set(pp, item, val)
                else:
                    log(f"can't apply mapping: {mapping}")
            elif mapping.fn == Fns.regex_groups:
                match_obj = mapping.source.col.regex_match(col)
                if len(match_obj.groups()) == len(mapping.target.items):
                    for val, item in zip(match_obj.groups(), mapping.target.items):
                        deep_set(pp, item, val)
                else:
                    log(f"can't apply mapping: {mapping}")

    models = [model_cls(**pp) for pp in pre_parsed]
    for i, m in enumerate(models):
        props = dict(m.dict())
        if props.get('model'):
            props['path_safe_model'] = props['model'].replace('/', '_').strip('.')
        else:
            props['path_safe_model'] = f'col:{i}'
        output = Path(opts.output.format(**props))
        if not output.exists():
            log(f"writing to {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(m.dict(), indent=2))
        else:
            log(f"not overwriting {output}")
    # log_pp([m.dict() for m in models])
    # log(f"{json.dumps(pre_parsed, indent=2)}")

if __name__ == '__main__':
    main(sys.argv[1:])
    # cp = CurlyPattern(sys.argv[1])
    # # print(cp.idx)
    # # print(list(cp.items(0, 0, len(cp.p))))
    # print(list(cp.expand()))

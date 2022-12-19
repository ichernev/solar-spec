import sys
from pathlib import Path
import json
import re
import yaml

class Matcher:
    def __init__(self, spec, pref):
        self.pref = pref
        self.row_h = spec['row_h']
        self.cell = spec.get('cell', r'(.*)')
        self._mline = None

    def match(self, line):
        if line[0]['text'] == self.row_h:
            self._mline = line

    def extr(self):
        if not self._mline:
            return []

        res = []
        for cell in self._mline[1:]:
            match = re.match(self.cell, cell['text'])
            if match:
                res.append(match.group(1))
        return res

    def __str__(self):
        return f"Matcher({self.pref}, {self.extr()})"

    @classmethod
    def extract(cls, spec, pref=''):
        res = []
        if 'row_h' in spec:
            res.append(cls(spec, pref))
        if isinstance(spec, dict):
            for k, v in spec.items():
                res.extend(cls.extract(v, pref + '.' + k))
        return res

def extract(inp, spec):
    matchers = Matcher.extract(spec)

    for block in inp:
        for line in block['data']:
            for matcher in matchers:
                matcher.match(line)

    for matcher in matchers:
        print(matcher)

def main(args):
    spec_f = args[0]
    spec = None
    with open(spec_f, 'rb') as f:
        spec = yaml.load(f, Loader=yaml.Loader)

    for pipe in spec['pipelines']:
        file = Path(spec_f).parent / pipe['file']
        file_rjson = str(file) + '.raw.json'
        extract(json.loads(Path(file_rjson).read_text()), pipe['data'])

if __name__ == '__main__':
    main(sys.argv[1:])

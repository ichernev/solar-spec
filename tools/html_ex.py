import requests
import bs4
from pathlib import Path
from collections import defaultdict
import json
import itertools
import functools
# import pprint
import argparse
import yaml
import sys

# url = "https://www.deyeinverter.com/product/hybrid-inverter-1/sun12-14-16ksg01lp1-1216kw-single-phase-3-mppt-hybrid-inverter.html"
# # req = requests.get(url)

# file = url.rsplit('/', 1)[-1]
# # (Path('cache') / file).write_text(req.text)

# body = (Path('cache') / file).read_text()
# soup = bs4.BeautifulSoup(body, features='html')

class Cell:
    def __init__(self, text, cs):
        self.text = text
        self.colspan = cs

    def __str__(self):
        return f"Cell({self.text}, {self.colspan})"

    def __repr__(self):
        return str(self)

def parse_table(table):
    # print("------ ------ ------")
    tbl = []
    for row in table.find_all('tr'):
        # print("-> ", end='')
        crow = []
        for cell in row.find_all('td'):
            cs = 1
            if cell.get('colspan'):
                cs = int(cell['colspan'])
            crow.append(Cell(cell.text.strip(), cs))

            # print(f"{cs}:{cell.text.strip()}", end=' ')
        tbl.append(crow)
        # print()
    return tbl

def parse_tables(soup):
    tbls = []
    for table in soup.find_all('table'):
        tbls.append(parse_table(table))
    return tbls

class Model:
    def __init__(self, name):
        self.name = name
        self.props = defaultdict(dict)

    def add_prop(self, section, key, value):
        self.props[section][key] = value

    def to_json(self):
        return {'name': self.name, 'props': dict(self.props.items())}

    def __str__(self):
        return f"Model({self.name})"

    def __repr___(self):
        return str(self)

def columnize(tables):
    models = []
    for cell in tables[0][0][1:]:
        models.append(Model(cell.text))

    section = 'none'
    print(models, file=sys.stderr)
    for row in itertools.chain(tables[0][1:], *tables[1:]):
        tcs = functools.reduce(lambda a, b: a + b.colspan, row, 0)
        if len(row) == 1:
            section = row[0].text
        elif tcs == 2:
            # compressed table - 2 columns
            assert len(row) == 2
            for m in models:
                m.add_prop(section, row[0].text, row[1].text)
        else:
            if tcs != len(models) + 1:
                print(f"{section} {row}", file=sys.stderr)
                continue
            # assert tcs == len(models) + 1
            key = row[0].text
            start_col = 0
            for cell in row[1:]:
                end_col = start_col + cell.colspan
                for col in range(start_col, end_col):
                    models[col].add_prop(section, key, cell.text)
    return models


def parse_args(args):
    parser = argparse.ArgumentParser("parse html spec tables")
    actions = parser.add_subparsers(dest='action')

    extract = actions.add_parser('extract')
    extract.add_argument('--slug', type=str,
                         help='name for cache/output (defaults to last part of url)')
    extract.add_argument('--cache-dir', type=str, default='cache',
                         help='where to save http cache')
    extract.add_argument('--json-dir', type=str, default='out',
                         help='where to store extracted json files, - means print')
    extract.add_argument('url', nargs=1, help="URL to extract")

    extract_yaml = actions.add_parser('extract-yaml')
    extract_yaml.add_argument('yaml', nargs=1, help="YAML file to parse")

    return parser.parse_args(args)

def main(args):
    opts = parse_args(args)

    if opts.action == 'extract':
        url = opts.url[0]
        # import pdb; pdb.set_trace()
        file = url.rsplit('/', 1)[-1]

        file_path = Path(opts.cache_dir) / file
        if not file_path.exists():
            req = requests.get(url)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(req.text)
        print(f"processing {file}", file=sys.stderr)
        body = file_path.read_text()
        soup = bs4.BeautifulSoup(body, features='html')
        tbls = parse_tables(soup)
        data = columnize(tbls)

        json_out = json.dumps([d.to_json() for d in data], indent=2)
        if opts.json_dir == '-':
            print(json_out)
        else:
            out_path = Path(opts.json_dir) / (file + '.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_out)
    elif opts.action == 'extract-yaml':
        cfg_path = Path(opts.yaml[0])
        cfg = yaml.load(cfg_path.read_bytes(), Loader=yaml.Loader)
        for item in cfg['pipelines']:
            if item.get('url'):
                args = ['extract',
                        '--cache-dir', cfg_path.parent / 'cache',
                        '--json-dir', 'gen',
                        item['url']]
                main([str(a) for a in args])

if __name__ == '__main__':
    main(sys.argv[1:])

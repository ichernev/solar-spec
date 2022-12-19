#!/usr/bin/env python3

import yaml
import json
import sys
import subprocess
from pathlib import Path

def cj(*stuff):
    if len(stuff) == 1 and isinstance(stuff, (list, tuple)):
        stuff = stuff[0]
    return ','.join(map(str, stuff))

def main(args):
    spec_f = args[0]

    spec = None
    with open(spec_f, 'r') as f:
        if spec_f.endswith('.json'):
            spec = json.load(f)
        elif spec_f.endswith('.yaml'):
            spec = yaml.safe_load(f)
        else:
            print("WTF")

    for pipe in spec['pipelines']:
        args = ["/home/iskren/src/solar/bin/tabula"]
        print(pipe)
        if pipe.get('area'):
            x1,y1,x2,y2 = pipe['area']
            args.extend(('-a', cj(y1, x1, y2, x2)))
        elif pipe.get('area_wh'):
            x,y,w,h = pipe['area_wh']
            args.extend(('-a', cj(y, x, y+h, x+w)))
        else:
            args.append('-g')
        if pipe.get('cols'):
            args.extend(('-c', ','.join(map(str, pipe['cols']))))
        if 'latice' in pipe:
            if pipe['latice'] == 0:
                args.append('-t')
            else:
                args.append('-l')
        args.extend(('-p', ','.join(map(str, pipe.get('pages', [1])))))
        args.extend(('-f', 'JSON'))
        file = Path(spec_f).parent / pipe['file']
        args.extend(('-o', str(file) + '.raw.json'))
        args.append(str(file))
        print(f"running {args}")
        subprocess.run(args)


if __name__ == '__main__':
    main(sys.argv[1:])


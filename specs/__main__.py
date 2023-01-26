import importlib
import json
import sys
import os
from pathlib import Path

SCHEMA_ROOT = Path(__file__).parent.parent / 'schemas'

items = {
    'solar/hybrid_inverter.schema': '.solar.hybrid_inverter.HybridInverter',
}


if __name__ == '__main__':
    for target, src in items.items():
        pkg, cls = src.rsplit('.', 1)
        src_pkg = importlib.import_module(pkg, package='specs')
        src_cls = getattr(src_pkg, cls)

        full_target = SCHEMA_ROOT / target
        os.makedirs(full_target.parent, exist_ok=True)
        with open(full_target, 'w') as f:
            print(f'{full_target}')
            json.dump(src_cls.schema(), f, indent=2)

#!/usr/bin/env python3

from argparse import ArgumentParser
import datetime
from pathlib import Path
import sys
import requests
import yaml
import os
import subprocess
import time
import json

def log(*args, file=sys.stderr):
    print(*args, file=file)

def parse(args):
    parser = ArgumentParser("Execute download/extract/map pipelines")

    parser.add_argument('--input-dir', type=str,
                        default="pipelines",
                        help="where to look for spec files")
    parser.add_argument('--filter', type=str,
                        help="a python expression that returns True/False given stem, group, stage")
    parser.add_argument('--output-pattern', type=str,
                        default="{stem}/{group}/{stage}/{filename}.{ext}",
                        help="output directory structure")
    parser.add_argument('--output-dir', type=str,
                        default="data",
                        help="output directory root (pattern is appended)")
    parser.add_argument('--force', action='store_true',
                        help="Ignore mtime and recompute everything")
    parser.add_argument('--ignore-errors', action='store_true',
                        help="Do not stop on the first encountered error (i.e by extractor/mapper)")

    return parser.parse_args(args)

def iter_tree(folder, suffix=None):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not suffix or file.endswith(suffix):
                yield Path(root) / file


def parse_yaml(file):
    body = Path(file).read_bytes()
    return yaml.load(body, Loader=yaml.Loader)


def recent_enough(file, interval=None, source=None):
    if interval:
        interval = (interval.total_seconds() if isinstance(interval, datetime.timedelta)
                    else interval)
        path = Path(file)
        return path.exists() and path.stat().st_ctime >= time.time() - interval
    else:
        return (Path(file).exists() and
                Path(file).stat().st_ctime >= Path(source).stat().st_ctime)


def merge(*dicts):
    res = {}
    for di in dicts:
        res.update(di)
    return res

def pass_filter(opts, props):
    return opts.filter is None or eval(opts.filter, {}, props)

def output_file(opts, props):
    """Sanitize group for now"""
    nprops = merge({}, props, {'group': props['group'].replace('/', '_')})
    return opts.output_pattern.format(**nprops)

def clean_stage(props, opts):
    import glob
    props = merge(props, {'filename': '*', 'ext': '*'})
    pat = opts.output_pattern.format(**props)
    for file in glob.glob(pat):
        Path(file).unlink()

def get_merged_stage_info(spec, pipe, key):
    specific_infos = pipe.get(key)
    if not specific_infos:
        return
    if not isinstance(specific_infos, list):
        specific_infos = [specific_infos]

    for specific_info in specific_infos:
        if specific_info and 'inherit' in specific_info:
            merged_info = merge(spec[key][specific_info['inherit']], specific_info)
            del merged_info['inherit']
        else:
            merged_info = specific_info
        yield merged_info

def out_path(opts, props, **extra):
    res = Path(opts.output_dir)
    props = merge(props, extra)
    try:
        res = res / props['stem']
        res = res / props['group']
        res = res / props['stage']
        if 'ext' in props:
            res = res / f"{props['filename']}{props['ext']}"
        else:
            res = res / f"{props['full_filename']}"
    except KeyError:
        pass
    return res

def stage_can_skip(opts, props, old_stage, new_stage):
    """
    If all files from previous stage (+ pipeline file) are older than all files
    from new stage, we can skip it.
    """
    last_mod = None
    if not out_path(opts, props, stage=old_stage):
        raise Exception("previous stage is empty")
    for file in out_path(opts, props, stage=old_stage).iterdir():
        if last_mod is None or last_mod < file.stat().st_mtime:
            last_mod = file.stat().st_mtime
    if last_mod is None:
        raise Exception("previous stage is empty")
    p_mtime = Path(props['pipeline_file']).stat().st_mtime
    if last_mod < p_mtime:
        last_mod = p_mtime

    first_mod = None
    if not out_path(opts, props, stage=new_stage).exists():
        return False
    for file in out_path(opts, props, stage=new_stage).iterdir():
        if first_mod is None or first_mod > file.stat().st_mtime:
            first_mod = file.stat().st_mtime

    return first_mod is not None and last_mod < first_mod

def download(item, spec, props, opts):
    # ext = {'datasheet': 'pdf', 'manual': 'pdf', 'image': None, 'page': 'html'}[item]
    ext = Path(item).suffix
    if not ext:
        # take it from URL
        ext = Path(spec).suffix
    else:
        # drop suffix from item
        item = Path(item).stem
    if not ext or not ext.startswith('.'):
        raise ValueError(f"Could not extract suffix from {item} or {spec}")
    ext = ext[1:] # skip the leading .
    props = merge(props, {'filename': item, 'ext': ext})
    target = Path(opts.output_dir) / output_file(opts, props)

    if recent_enough(target, interval=datetime.timedelta(weeks=12)):
        log(f"using existing {target}")
        return

    assert isinstance(spec, str)
    res = requests.get(spec)
    os.makedirs(target.parent, exist_ok=True)
    target.write_bytes(res.content)
    log(f"downloaded {spec} to {target}")

def download_stage(stage_info, group_props, opts):
    group_props = merge(group_props, {'stage': 'download'})

    if not pass_filter(opts, group_props):
        return

    if stage_info is None:
        return

    # We try to re-use artefacts, so don't wipe them
    # clean_stage(group_props, opts)
    for item, loc in stage_info.items():
        download(item, loc, group_props, opts)

def _run(opts, args):
    res = subprocess.run(args)
    if res.returncode != 0 and not opts.ignore_errors:
        raise Exception(f"Failed during execution of {' '.join([str(a) for a in args])}")
    return res

def extract_stage(stage_info, group_props, opts):
    pdf_ex = Path(__file__).parent / 'pdf_ex.py'
    group_props = merge(group_props, {'stage': 'extract'})

    if stage_info is None:
        return

    if not pass_filter(opts, group_props):
        return

    log(f"stage_info: {stage_info}")
    log(f"group_gprops: {group_props}")
    clean_stage(group_props, opts)
    source = stage_info.pop('source')
    source, ext = source.rsplit('.', 1)
    source_file = Path(opts.output_dir) / output_file(opts, merge(
        group_props, {'stage': 'download', 'filename': source, 'ext': ext}))
    target_file = Path(opts.output_dir) / output_file(opts, merge(
        group_props, {'filename': source, 'ext': 'csv'}))
    stage_info['file'] = str(source_file)
    stage_info['out'] = str(target_file)

    if not opts.force and recent_enough(target_file, source=source_file):
        log(f"skipping {group_props['stem']} {group_props['group']} extract")
        return

    _run(opts, [
        pdf_ex,
        'extract-cfg',
        '--config', json.dumps(stage_info, separators=(',', ':'))
    ])

def map_stage(stage_info, group_props, opts):
    if stage_info is None:
        return

    mapper = Path(__file__).parent / 'mapper.py'
    group_props = merge(group_props, {'stage': 'map'})

    if not pass_filter(opts, group_props):
        return

    if (not opts.force and
            stage_can_skip(opts, group_props, old_stage='extract', new_stage='map')):
        log(f"skipping {group_props['stem']} {group_props['group']} map")
        return

    log(f"stage_info: {stage_info}")
    log(f"group_gprops: {group_props}")
    clean_stage(group_props, opts)
    source = stage_info.pop('source')
    source, ext = source.rsplit('.', 1)
    source_file = Path(opts.output_dir) / output_file(opts, merge(
        group_props, {'stage': 'extract', 'filename': source, 'ext': ext}))
    target_file = Path(opts.output_dir) / output_file(opts, merge(
        group_props, {'filename': '{path_safe_model}', 'ext': 'json'}))
    # stage_info['file'] = str(source_file)
    # stage_info['out'] = str(target_file)
    _run(opts, [
        # '/usr/bin/echo',
        str(mapper),
        '--input', source_file,
        '--output', target_file,
        '--schema', stage_info.pop('schema'),
        '--mapper-spec-json', json.dumps(stage_info, separators=(',', ':')),
    ])

def main(args):
    opts = parse(args)

    for pipe_file in iter_tree(opts.input_dir, suffix=".yaml"):
        # drop .yaml suffix
        stem = str(Path(str(pipe_file)[:-5]).relative_to(opts.input_dir))
        pipes = parse_yaml(pipe_file)

        log(f"looking at {stem}.yaml")
        props = pipes['props']
        props['stem'] = str(stem)
        props['pipeline_file'] = pipe_file

        for pipe in pipes['pipelines']:
            gprops = merge(props, pipe['meta'])
            gprops['group'] = pipe['meta']['group'].format(**gprops)
            download_stage(pipe.get('download'), gprops, opts)
            for stage_info in get_merged_stage_info(pipes, pipe, 'extractor'):
                extract_stage(stage_info, gprops, opts)
            for stage_info in get_merged_stage_info(pipes, pipe, 'mapper'):
                map_stage(stage_info, gprops, opts)


if __name__ == '__main__':
    main(sys.argv[1:])

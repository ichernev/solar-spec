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


def merge(*dicts):
    res = {}
    for di in dicts:
        res.update(di)
    return res


def download(item, spec, props, opts):
    ext = {'datasheet': 'pdf', 'manual': 'pdf', 'image': None, 'page': 'html'}[item]
    if ext is None:
        # take it from URL
        _, ext = spec.rsplit('.', 1)
    props = merge(props, {'filename': item, 'ext': ext})
    target = Path(opts.output_dir) / opts.output_pattern.format(**props)

    if recent_enough(target, interval=datetime.timedelta(weeks=1)):
        log(f"using existing {target}")
        return

    assert isinstance(spec, str)
    res = requests.get(spec)
    os.makedirs(target.parent, exist_ok=True)
    target.write_bytes(res.content)
    log(f"downloaded {spec} to {target}")

def clean_stage(props, opts):
    import glob
    props = merge(props, {'filename': '*', 'ext': '*'})
    pat = opts.output_pattern.format(**props)
    for file in glob.glob(pat):
        Path(file).unlink()

def download_stage(stage_info, group_props, opts):
    if stage_info is None:
        return

    group_props = merge(group_props, {'stage': 'download'})
    # We try to re-use artefacts, so don't wipe them
    # clean_stage(group_props, opts)
    for item, loc in stage_info.items():
        if item in ('datasheet', 'manual', 'image'):
            # log(f"gprops: {group_props}")
            download(item, loc, group_props, opts)


def get_merged_stage_info(spec, pipe, key):
    specific_info = pipe.get(key)
    if specific_info and 'inherit' in specific_info:
        merged_info = merge(spec[key][specific_info['inherit']], specific_info)
        del merged_info['inherit']
    else:
        merged_info = specific_info
    return merged_info

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
    for file in out_path(opts, props, stage=old_stage).iterdir():
        if last_mod is None or last_mod < file.stat().st_mtime:
            last_mod = file.stat().st_mtime
    if last_mod is None:
        raise Exception("previous stage is empty")
    p_mtime = Path(props['pipeline_file']).stat().st_mtime
    if last_mod < p_mtime:
        last_mod = p_mtime

    first_mod = None
    for file in out_path(opts, props, stage=new_stage).iterdir():
        if first_mod is None or first_mod > file.stat().st_mtime:
            first_mod = file.stat().st_mtime

    return first_mod is not None and last_mod < first_mod

def extract_stage(stage_info, group_props, opts):
    pdf_ex = Path(__file__).parent / 'pdf_ex.py'

    if (not opts.force and
            stage_can_skip(opts, group_props, old_stage='download', new_stage='extract')):
        log(f"skipping {group_props['stem']} {group_props['group']} extract")
        return

    log(f"stage_info: {stage_info}")
    log(f"group_gprops: {group_props}")
    clean_stage(merge(group_props, {'stage': 'extract'}), opts)
    source = stage_info.pop('source')
    source, ext = source.rsplit('.', 1)
    source_file = Path(opts.output_dir) / opts.output_pattern.format(**merge(
        group_props, {'stage': 'download', 'filename': source, 'ext': ext}))
    target_file = Path(opts.output_dir) / opts.output_pattern.format(**merge(
        group_props, {'stage': 'extract', 'filename': source, 'ext': 'csv'}))
    stage_info['file'] = str(source_file)
    stage_info['out'] = str(target_file)
    subprocess.run([str(pdf_ex), 'extract-cfg', '--config', json.dumps(stage_info, separators=(',', ':'))])

def map_stage(stage_info, group_props, opts):
    mapper = Path(__file__).parent / 'mapper.py'

    if (not opts.force and
            stage_can_skip(opts, group_props, old_stage='extract', new_stage='map')):
        log(f"skipping {group_props['stem']} {group_props['group']} map")
        return

    log(f"stage_info: {stage_info}")
    log(f"group_gprops: {group_props}")
    clean_stage(merge(group_props, {'stage': 'map'}), opts)
    source = stage_info.pop('source')
    source, ext = source.rsplit('.', 1)
    source_file = Path(opts.output_dir) / opts.output_pattern.format(**merge(
        group_props, {'stage': 'extract', 'filename': source, 'ext': ext}))
    target_file = Path(opts.output_dir) / opts.output_pattern.format(**merge(
        group_props, {'stage': 'map', 'filename': '{path_safe_model}', 'ext': 'json'}))
    # stage_info['file'] = str(source_file)
    # stage_info['out'] = str(target_file)
    subprocess.run([
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
            extract_stage(get_merged_stage_info(pipes, pipe, 'extractor'), gprops, opts)
            map_stage(get_merged_stage_info(pipes, pipe, 'mapper'), gprops, opts)


if __name__ == '__main__':
    main(sys.argv[1:])

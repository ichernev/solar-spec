#!/usr/bin/env python3

import pdfminer
import pdfminer.high_level
import pdfminer.layout
import io
import sys
import math
import bs4
import itertools
import functools
from collections import defaultdict
import argparse
import json
from pathlib import Path
import yaml
import os

from pydantic import BaseModel, Field
from typing import NamedTuple
from enum import Enum

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(filename)s:%(lineno)d - %(message)s')
sh.setFormatter(formatter)
log.addHandler(sh)

class BBoxHelper(object):

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls(x, y, x+w, y+h)

    @classmethod
    def make_span(cls, frto, axis):
        if axis == 'x':
            return cls(frto[0], -math.inf, frto[1], +math.inf)
        elif axis == 'y':
            return cls(-math.inf, frto[0], +math.inf, frto[1])
        raise ValueError(f"Unknown axis {axis}")

    def __init__(self, a, b=None, c=None, d=None):
        if b is not None and c is not None and d is not None:
            args = [a, b, c, d]
        else:
            args = map(float, a.split(','))
        self.minx, self.miny, self.maxx, self.maxy = args

    def to_json(self):
        return f"{self.minx},{self.miny},{self.maxx},{self.maxy}"

    def contains(self, other):
        return (self.minx <= other.minx and other.maxx <= self.maxx and
                self.miny <= other.miny and other.maxy <= self.maxy)

    def intersects(self, other):
        return (max(self.minx, other.minx) < min(self.maxx, other.maxx) and
                max(self.miny, other.miny) < min(self.maxy, other.maxy))

    def grow(self, amt):
        if isinstance(amt, int) or isinstance(amt, float):
            amt = (amt,)
        if len(amt) == 1:
            amt = (amt[0], amt[0])
        if len(amt) == 2:
            amt = (amt[0], amt[1], amt[0], amt[1])
        return BBoxHelper(self.minx - amt[0], self.miny - amt[1], self.maxx + amt[2], self.maxy + amt[3])

    def accomodating(self, other):
        return BBoxHelper(
            min(self.minx, other.minx),
            min(self.miny, other.miny),
            max(self.maxx, other.maxx),
            max(self.maxy, other.maxy))

    def rel_to(self, containing, direction):
        if direction == 'right':
            r = BBoxHelper(self.maxx, self.miny, containing.maxx, self.maxy)
        elif direction == 'left':
            r = BBoxHelper(containing.minx, self.miny, self.minx, self.maxy)
        elif direction == 'up':
            r = BBoxHelper(self.minx, self.maxy, self.maxx, containing.maxy)
        elif direction == 'down':
            r = BBoxHelper(self.minx, containing.miny, self.maxx, self.miny)
        else:
            raise Exception('unknown direction')

        return r

    @property
    def midx(self):
        return (self.minx + self.maxx) / 2

    @property
    def midy(self):
        return (self.miny + self.maxy) / 2

    def set_minx(self, minx):
        return BBoxHelper(minx, self.miny, self.maxx, self.maxy)

    def set_miny(self, miny):
        return BBoxHelper(self.minx, miny, self.maxx, self.maxy)

    def set_maxx(self, maxx):
        return BBoxHelper(self.minx, self.miny, maxx, self.maxy)

    def set_maxy(self, maxy):
        return BBoxHelper(self.minx, self.miny, self.maxx, self.maxy)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.minx, self.miny, self.maxx, self.maxy)

BBoxHelper.COVERALL = BBoxHelper(-math.inf, -math.inf, math.inf, math.inf)
BBoxHelper.EMPTY = BBoxHelper(+math.inf, +math.inf, -math.inf, -math.inf)

class TextHelper(object):
    def __init__(self, text_tag):
        self.text = text_tag.text
        self.bbox = BBoxHelper(text_tag['bbox'])

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, repr(self.text), repr(self.bbox))

class TextLineHelper(object):
    @classmethod
    def merge(cls, a, b, sep=' '):
        assert a.page == b.page
        return cls(
            a.text + sep + b.text,
            a.bbox.accomodating(b.bbox),
            a.page if a.page == b.page else None)

    @classmethod
    def from_tag(cls, tl_tag, page):
        pieces = [[]]
        for text_tag in tl_tag.contents:
            if not isinstance(text_tag, bs4.element.Tag):
                # superflous whitespace between tags
                continue
            if text_tag.get('font'):
                pieces[-1].append(text_tag)
            else:
                pieces.append([])
        tags = []
        for piece in pieces:
            if piece:
                bbox = BBoxHelper.EMPTY
                text = []
                # strip
                while len(piece) and (not piece[0].string or piece[0].string.isspace()):
                    piece.pop(0)
                while len(piece) and (not piece[-1].string or piece[-1].string.isspace()):
                    piece.pop(-1)
                for letter in piece:
                    bbox = bbox.accomodating(BBoxHelper(letter['bbox']))
                    text.append(letter.string)
                text = [l or ' ' for l in text]
                tags.append(cls(''.join(text), bbox, page))
        # import pdb; pdb.set_trace()
        return tags

    def __init__(self, *args):
        if len(args) == 2:
            text_line_tag, page = args
            assert isinstance(text_line_tag, bs4.element.Tag)
            self.text = "".join(map(lambda t: t.text, text_line_tag.find_all('text'))).rstrip('\n')
            self.bbox = BBoxHelper(text_line_tag['bbox'])
            self.page = page
        elif len(args) == 3:
            self.text, self.bbox, self.page = args
        else:
            raise Exception("Expected 2 or 3 args, got {}".format(len(args)))

        self.adjy = None

    def rel_find_one(self, **kwargs):
        """
        Return the closest match from rel_find
        """
        ret = kwargs.pop('ret', 'text')
        direction = kwargs.get('direction') or 'left'
        results = self.rel_find(ret='textline', **kwargs)

        # results.sort(key=lambda tl: tl.bbox.minx if direction in ['left', 'right'] else tl.bbox.miny)
        # res = results[0 if direction in ['left', 'up'] else -1]
        res = results[0]

        if ret == 'text':
            return res.text
        elif ret == 'match':
            m = re.match(kwargs['pattern'] or '.*', res.text)
            if len(m.groups()) == 1:
                return m.group(1)
            else:
                return m.group(0)
        else:
            return res

    def rel_find(self, **kwargs): # pattern='.*', direction='left', grow=0, ret='text', bbox_match='contains'):
        page_bbox = self.page.bbox
        direction = kwargs.pop('direction', 'left')
        grow = kwargs.pop('grow', 0)
        target_bbox = self.bbox.rel_to(page_bbox, direction).grow(grow)
        kwargs['sort'] = direction
        return self.page.find(bbox=target_bbox, **kwargs) #pattern=pattern, ret=ret, bbox_match=bbox_match)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, repr(self.text), repr(self.bbox), self.page._idx)

class PageHelper(object):

    LINE_EPS = 2
    WORD_EPS = 5
    MAP = {
        '\xa0': ' ',
    }

    def __init__(self, idx, xml):
        self._id = xml['id']
        self._idx = idx
        self._xml = xml
        self.bbox = BBoxHelper(self._xml['bbox'])
        self._textlines = list(itertools.chain(*(TextLineHelper.from_tag(tl, self) for tl in self._xml.find_all('textline'))))
        # self._textlines.extend(self._textlineify(self._xml))
        self._flip_y(self.bbox.maxy)
        self._drop_crap()

    def _drop_crap(self):
        self._textlines = list(filter(lambda tl: tl.text, self._textlines))

    def _flip_y(self, p_maxy):
        for tl in self._textlines:
            miny, maxy = tl.bbox.miny, tl.bbox.maxy
            tl.bbox.miny, tl.bbox.maxy = p_maxy - maxy, p_maxy - miny

    def find(self, bbox=BBoxHelper.COVERALL, pattern='.*', ret='text', bbox_match='contains', sort=None):
        results = []
        for tl in self._textlines:
            if getattr(bbox, bbox_match)(tl.bbox):
                m = re.search(pattern, tl.text)
                if m:
                    if ret == 'text':
                        x = tl.text
                    elif ret == 'match':
                        gr = m.groups()
                        if len(gr) == 1:
                            # first group
                            x = gr[0]
                        else:
                            # full match
                            x = m.group(0)
                    elif ret == 'groups':
                        x = m.groups()
                    elif ret == 'textline':
                        x = tl
                    else:
                        raise Exception("unknown return type {}".format(ret))

                    results.append((x, tl) if sort else x)
        if sort:
            results.sort(key=lambda x_tl: ((x_tl[1].bbox.minx if sort in ['left', 'right'] else x_tl[1].bbox.miny) *
                                           (1 if sort in ['up', 'right'] else -1)))
            results = list(map(lambda x_tl: x_tl[0], results))
        return results

    def matches_idx(self, idx):
        if idx is None:
            return True
        elif isinstance(idx, list) or isinstance(idx, range):
            return self._idx in idx
        else:
            return self._idx == idx

    def _textlineify(self, xml):
        texts = []
        for text in xml.find_all('text'):
            if text.parent.name == 'textline':
                continue
            texts.append(TextHelper(text))

        # print("{}".format("\n".join(map(lambda t: repr(t), texts[:100]))))

        if not texts:
            return []
        # return []

        texts.sort(key=lambda t: -t.bbox.miny)
        # text_lines = list(map(lambda tl: TextLineHelper(tl, None), self._xml.find_all('textline')))
        # text_lines.sort(key=lambda tl: tl.bbox.miny)

        begi = 0
        lasty = texts[0].bbox.miny
        for i in range(1, len(texts) + 1):
            lasty = texts[i-1].bbox.miny
            cury = texts[i].bbox.miny if i < len(texts) else None
            if i == len(texts) or abs(cury - lasty) > self.LINE_EPS:
                sumy = sum(map(lambda j: texts[j].bbox.miny, range(begi, i)))
                avgy = sumy / (i - begi)
                for j in range(begi, i):
                    texts[j].adjy = avgy

                begi = i


        texts.sort(key=lambda tl: (-tl.adjy, tl.bbox.minx))
        begi = 0
        tspans = []
        ctxt = self._fix_txt(texts[0].text)
        cbbox = texts[0].bbox
        for i in range(1, len(texts) + 1):
            if i == len(texts) or texts[i-1].adjy != texts[i].adjy or texts[i].bbox.minx - texts[i-1].bbox.maxx > self.WORD_EPS:
                if ctxt.strip():
                    tspans.append(TextLineHelper(ctxt.strip(), cbbox, self))

                if i < len(texts):
                    begi = i
                    ctxt = self._fix_txt(texts[i].text)
                    cbbox = texts[i].bbox
            else:
                ctxt += self._fix_txt(texts[i].text)
                cbbox = cbbox.accomodating(texts[i].bbox)

        # print("{}".format(tspans))
        return tspans

    def _fix_txt(self, txt):
        if txt in self.MAP:
            return self.MAP[txt]
        return txt

    def __repr__(self):
        return "page {}:\n".format(self._idx) + "\n".join(map(repr, self._textlines))


class PDFDoc(object):

    @classmethod
    def from_file(cls, filename):
        xml = extract_pdfminer_xml(filename)
        return cls(xml)

    def __init__(self, xml_str):
        self._xml = bs4.BeautifulSoup(xml_str, 'xml')
        self._pages = []
        for idx, page_xml in enumerate(self._xml.find_all('page')):
            self._pages.append(PageHelper(idx, page_xml))
            # self._pages.append(list(map(lambda tl: TextLineHelper(tl, page=i),
            #                             page.find_all('textline'))))

    # ret = text|match|textline
    def find(self, **kwargs):
        res = []
        page = kwargs.pop('page', None)
        for p in self._pages:
            if p.matches_idx(page):
                res.extend(p.find(**kwargs))

        return res

    def find_one(self, **kwargs):
        res = self.find(**kwargs)
        assert(len(res) == 1)
        return res[0]

    def find_first(self, **kwargs):
        assert 'sort' in kwargs
        res = self.find(**kwargs)
        assert len(res) > 0
        return res[0]

    def page(self, idx):
        return self._pages[idx]

    def __repr__(self):
        return "pdfdoc:\n" + "\n\n".join(map(repr, self._pages)) + "\n"


def extract_pdfminer_text(path):
    with open(path, 'rb') as fp:
        # outfp = io.BytesIO()
        outfp = io.StringIO()
        laparams = pdfminer.layout.LAParams()
        pdfminer.high_level.extract_text_to_fp(fp, output_type="text", outfp=outfp, laparams=laparams)
        return outfp.getvalue()

def extract_pdfminer_xml(path):
    with open(path, 'rb') as fp:
        outfp = io.BytesIO()
        laparams = pdfminer.layout.LAParams()
        pdfminer.high_level.extract_text_to_fp(fp, output_type="xml", outfp=outfp, laparams=laparams)
        return outfp.getvalue().decode('utf-8')

def _cluster(items, getter, eps, eps_rel, sort):
    items = list(items)
    if sort:
        items.sort(key=getter)
    # clusters = []
    beg = 0
    for i in range(1, len(items)):
        val = getter(items[i])
        if ((eps is None and val != getter(items[i-1])) or
            (eps is not None and abs(val - getter(items[i-1 if eps_rel else beg])) > eps)
        ):
            yield list(items[beg:i])
            beg = i
    # clusters.append(list(items[beg:]))
    yield list(items[beg:])
    # return clusters


identity = lambda x: x

def cluster(items, getter=identity, eps=None, sort=True, eps_rel=True, as_list=True):
    """
    group items by getter, fuzzy by epsilon, compare epsilon to first in group
    (absolute) or previous (relative).
    """
    it = _cluster(items, getter, eps=eps, eps_rel=eps_rel, sort=sort)
    if as_list:
        return list(it)
    return it


def _merge(items, condition):
    grp = []
    for item in items:
        if len(grp) == 0 or condition(grp[-1], item):
            grp.append(item)
        else:
            yield grp
            grp = [item]
    yield grp


def merge(items, condition, as_list=True):
    it = _merge(items, condition)
    if as_list:
        return list(it)
    return it


class Column:
    def __init__(self, idx):
        self.idx = idx
        self.attr = defaultdict(dict)

    def add_attribute(self, section, key, val):
        if self.attr[section].get(key):
            self.attr[section][key] += f' {val}'
        else:
            self.attr[section][key] = f'{val}'

    def __str__(self):
        return f"Column({self.idx}, #{len(self.attr)})"

    def to_json(self):
        return {'idx': self.idx, 'props': dict(self.attr)}


class Tabulate:

    # Header row items must be this close to each other to be part of the header row
    _HEADER_EPS = 10
    _MATCH_EPS = 10
    # Maximum difference between odd/even rows for fine adjustment
    _SPAN_ADJ_EPS = 20
    # Max outside of row/col span for cell content to be considered inside
    _SPAN_MATCH_EPS = 5
    # max difference between midx for different cells in the same row and column
    _CLUSTER_COL_EPS = 5
    # how far can be the computed column edge from cell center to actual column
    # edge (computed by header cell centers)
    _COL_EDGE_EPS = 15

    def __init__(self, pdf, config):
        self.pdf = pdf
        self.page_idx = config.page - 1 # idx is 0 based
        self.area = BBoxHelper.from_xywh(*config.area)
        self.config = config

        self.textlines = list(filter(lambda tl: self.area.contains(tl.bbox), self.pdf.page(self.page_idx)._textlines))
        self._tl_trash = []

        # self._rows = defaultdict(list)
        # self._cols = defaultdict(list)

    @property
    def fix(self):
        return self.config.fix

    @staticmethod
    def _keyer(prop, extr='min'):
        def key(tl):
            val = getattr(tl.bbox, prop)
            if extr == 'max':
                val *= -1
            return val
        return key

    def _find_first(self, prop, extr='min'):
        tls = list(self.textlines)
        # def key(tl):
        #     val = getattr(tl.bbox, prop)
        #     if extr == 'max':
        #         val *= -1
        #     return val
        tls.sort(key=self._keyer(prop, extr))

        orig = getattr(tls[0].bbox, prop)
        group = list(itertools.takewhile(lambda tl: abs(getattr(tl.bbox, prop)-orig) < self._HEADER_EPS, tls))
        sort_prop = prop[:-1] + ('x' if prop[-1] == 'y' else 'y')
        group.sort(key=self._keyer(sort_prop, 'min'))
        return group

    def _match_seq(self, seq, prop, tl, cid):
        best = seq[0]
        key = self._keyer(prop)
        for item in seq[1:]:
            if abs(key(item) - key(tl)) < abs(key(best) - key(tl)):
                best = item

        if abs(key(best) - key(tl)) < self._MATCH_EPS:
            return best._cell[cid]
        return None

    def _get_span(self, item, axis):
        if axis == 'x':
            return BBoxHelper(item._span[0], -math.inf, item._span[1], +math.inf)
        elif axis == 'y':
            return BBoxHelper(-math.inf, item._span[0], +math.inf, item._span[1])

    def _match_span_multi(self, seq, tl, axis):
        res = []
        for item in seq:
            if self._get_span(item, axis).contains(tl.bbox):
                res.append(item._cell[1 if axis == 'x' else 0])
        return res


    def _match_span(self, seq, tl, axis):
        for item in seq:
            if self._get_span(item, axis)\
                    .contains(tl.bbox):
                #.grow(self._SPAN_MATCH_EPS)
                return item._cell[1 if axis == 'x' else 0]
        return None

    # def _set_row(self, tl, row):
    #     old_row = None
    #     if tl._cell is None:
    #         tl._cell = (row, None)
    #     else:
    #         old_row = tl._cell[0]
    #         tl._cell = (row, tl._cell[1])

    #     if old_row is not None:
    #         rowl = self._rows[old_row]
    #         rowl.pop(rowl.index(tl))
    #     if row is not None:
    #         self._rows[row].append(tl)

    # def _set_col(self, tl, col):
    #     old_col = None
    #     if tl._cell is None:
    #         tl._cell = (None, col)
    #     else:
    #         old_col = tl._cell[1]
    #         tl._cell = (tl._cell[0], col)

    #     if old_col is not None:
    #         coll = self._cols[old_col]
    #         coll.pop(coll.index(tl))
    #     if col is not None:
    #         self._cols[col].append(tl)

    # def _set_row_col(self, tl, row, col):
    #     self._set_row(tl, row)
    #     self._set_col(tl, col)

    # def _get_row(self, tl):
    #     if tl._cell is None:
    #         return None
    #     return tl_cell[0]

    # def _get_col(self, tl):
    #     if tl._cell is None:
    #         return None
    #     return tl_cell[1]

    def _alloc_spans_raw(self, seq, axis, end):
        for i, item in enumerate(reversed(seq)):
            mid = getattr(item.bbox, f'mid{axis}')
            beg = 2 * mid - end # mid - (end - mid)
            item._span = (beg, end)
            end = beg

    def _alloc_spans(self, seq, axis, adj=None):
        end = getattr(self.area, f'max{axis}')
        self._alloc_spans_raw(seq, axis, end)

        if adj is not None:
            # even, odd, from end
            lens = [[], []]
            for i, item in enumerate(reversed(seq)):
                lens[i%2].append(item._span[1] - item._span[0])

            lens[0].sort()
            lens[1].sort()
            a = lens[0][len(lens[0]) // 2]
            b = lens[1][len(lens[1]) // 2]
            if abs(a - b) < self._SPAN_ADJ_EPS:
                adj_end = end - (a - b) / 4
                print(f"adjusting {axis} by {(a - b) / 4}", file=sys.stderr)
                self._alloc_spans_raw(seq, axis, adj_end)

    def _alloc_spans_nbr(self, seq, axis):
        beg = getattr(self.area, f'min{axis}')
        end = getattr(self.area, f'max{axis}')
        for i, item in enumerate(seq):
            start = getattr(seq[i-1].bbox, f'mid{axis}') if i > 1 else beg
            stop  = getattr(seq[i+1].bbox, f'mid{axis}') if i + 1 < len(seq) else end
            item._span = (start, stop)

    # def _fix_colspan(self):
    #     for row in self._rows:
    #         by_cell = self._group_by_cell(row, 1)
    #         if len(by_cell) == len(self.col_h):
    #             # this is done
    #             continue
    #         else:
    #             # group by mids (with some EPS)
    #             # find colspan edges with alloc algo
    #             # match edges with real col edges
    #             pass#

    def _find_col_edge(self, col_edge):
        for cid, col in enumerate(self._header_row):
            if abs(col._tmp['span_x'][0] - col_edge) < self._COL_EDGE_EPS:
                return cid
        return None

    def _merge_multiline(self, seq):
        # Merge multiline header lines
        def pn_diff(prev, next):
            return next.bbox.miny - prev.bbox.maxy
        diffs = [pn_diff(prev, next)
                 for prev, next in zip(seq, seq[1:])]
        min_diffs = next(cluster(diffs, eps=2, eps_rel=False, as_list=False))

        res = []
        for merged in merge(seq, condition=lambda p, n: pn_diff(p, n) < min_diffs[-1]+0.001, as_list=True):
            res.append(self._merge_cells(merged))

        return res

    def _merge_cells(self, seq, axis='y', upd_source=True, joiner=' '):
        seq = list(seq)
        if len(seq) == 1:
            return seq[0]

        seq.sort(key=self._keyer(f'mid{axis}'))
        mitem = functools.reduce(lambda a, b: TextLineHelper.merge(a, b, joiner), seq)
        mitem._tmp = dict(seq[0]._tmp)
        if upd_source:
            for item in seq:
                self.textlines.remove(item)
                self._tl_trash.append(item)
            self.textlines.append(mitem)
        return mitem

    def _get_row_header(self):
        cols = cluster(self.textlines, lambda tl: tl.bbox.minx, eps=5, eps_rel=False, sort=True, as_list=False)
        first_col = next(cols)
        first_col.sort(key=self._keyer('midy'))

        # remove top-left cell
        # crap = first_col[0]
        # crap._tmp['is_header'] = True
        # first_col = first_col[1:]

        # log(f"first_col {first_col}")
        if self.config.sections == 'first-col':
            second_col = next(cols)
            second_col.sort(key=self._keyer('midy'))

            section_col = first_col
            header_col = second_col
        else:
            section_col = None
            header_col = first_col

        if self.config.multiline_row_header:
            header_col = self._merge_multiline(header_col)

        if self.config.sections == 'table-center':
            section_col = list(filter(lambda tl: tl._tmp.get('is_section'), self.textlines))
            section_col.sort(key=self._keyer('midy'))
            # assign phantom rows, so spans work
            for tl in section_col:
                phantom = TextLineHelper('', tl.bbox, tl.page)
                phantom._tmp = {'is_phantom': True}
                header_col.append(phantom)
            header_col.sort(key=self._keyer('midy'))

        if section_col:
            if self.config.multiline_section_header:
                section_col = self._merge_multiline(section_col)

            # Assign sections
            for i, tl in enumerate(section_col):
                tl._tmp['is_header'] = True
                tl._tmp['is_section'] = True
                tl._tmp['row_id'] = i

            # TODO: check why sections are not picked up for table-center
            sid = 0
            mid_eps = 2
            for tl in header_col:
                # log(f"looking at row {tl.text}")
                if (sid < len(section_col) and
                    tl.bbox.midy >= section_col[sid].bbox.midy - mid_eps
                ):
                    sid += 1
                # log(f"-- assign {sid-1}")
                tl._tmp['section_id'] = sid - 1

        for i, tl in enumerate(header_col):
            tl._tmp['is_header'] = True
            tl._tmp['row_id'] = i

        self._header_col = header_col
        self._header_col_map = {item._tmp['row_id']: item for item in self._header_col}
        self._section_col = section_col

        for head in self._header_col:
            log.info(f"H {head.text} {head._tmp.get('section_id', 'none')}")

        return self._header_col

    def _get_col_header(self):
        rows = cluster(self._non_header_tls(),
                       lambda tl: tl.bbox.midy, eps=2, eps_rel=False, sort=True, as_list=False)
        first_row = next(rows)
        first_row.sort(key=self._keyer('midx'))
        for i, tl in enumerate(first_row):
            tl._tmp['is_header'] = True
            tl._tmp['col_id'] = i

        self._set_span_mids(first_row, 'x')

        self._header_row = first_row
        for item in first_row:
            log.info(f"{item.text} {item._tmp['span_x']}")
        return self._header_row

    def _non_header_tls(self):
        return filter(lambda tl: not tl._tmp.get('is_header'), self.textlines)

    def _row_certain(self):
        return filter(lambda tl: 'row_id' in tl._tmp, self._non_header_tls())

    def _row_col_certain(self):
        return filter(lambda tl: 'col_ids' in tl._tmp, self._row_certain())

    def _set_span_nbr(self, seq, axis):
        beg = getattr(self.area, f'min{axis}')
        end = getattr(self.area, f'max{axis}')
        # seq = self._header_col
        for i, item in enumerate(seq):
            start = getattr(seq[i-1].bbox, f'mid{axis}') if i > 1 else beg
            stop  = getattr(seq[i+1].bbox, f'mid{axis}') if i + 1 < len(seq) else end
            item._tmp[f'span_{axis}'] = (start, stop)

    def _set_span_mids(self, seq, axis):
        end = getattr(self.area, f'max{axis}')
        for item in reversed(seq):
            beg = 2 * getattr(item.bbox, f'mid{axis}') - end
            item._tmp[f'span_{axis}'] = (beg, end)
            end = beg

    def _match_span_multi2(self, seq, axis, tl):
        res = []
        for item in seq:
            if BBoxHelper.make_span(item._tmp[f'span_{axis}'], axis).contains(tl.bbox):
                res.append(item._tmp['row_id' if 'y' else 'col_id'])
        return res

    def _match_bounds(self, bbox, bounds, axis):
        for i in range(1, len(bounds)):
            if (bounds[i-1] and bounds[i] and
                BBoxHelper.make_span((bounds[i-1], bounds[i]), axis).contains(bbox)
            ):
                return i-1
        return None

    def _partition_rows(self):
        row_h = self._header_col
        self._set_span_nbr(self._header_col, 'y')

        # for item in self._header_col[:10]:
        #     .inf.infoolog(f"ZZ {item.text} {item._tmp['span_y']}")

        disputed = defaultdict(int)
        row_certain = []
        for tl in self._non_header_tls():
            grp = self._match_span_multi2(self._header_col, 'y', tl)
            if len(grp) == 1:
                tl._tmp['row_id'] = grp[0]
                row_certain.append(tl)
            else:
                # log(f"-- dispute {tl} {grp}")
                tl._tmp['row_ids'] = grp
                for gi in grp:
                    disputed[gi] += 1
        # log(f"disputed {disputed}")

        # compute known row bounds
        bounds = [None] * (len(self._header_col) + 1)
        for i in range(1, len(self._header_col)):
            if i-1 not in disputed and i not in disputed:
                bounds[i] = (self._header_col[i-1].bbox.midy +
                             self._header_col[i].bbox.midy) / 2

        brev = lambda i, b: 2 * row_h[i].bbox.midy - b[i+1]
        bfwd = lambda i, b: 2 * row_h[i-1].bbox.midy - b[i-1]
        bounds[0] = brev(0, bounds) if bounds[1] is not None else None # self.area.miny
        bounds[-1] = bfwd(len(bounds)-1, bounds) if bounds[-2] is not None else None # self.area.maxy
        fwd_bounds = list(bounds)
        for i in range(1, len(fwd_bounds)):
            if fwd_bounds[i] is None and fwd_bounds[i-1] is not None:
                fwd_bounds[i] = bfwd(i, fwd_bounds)
        rev_bounds = list(bounds)
        for i in range(len(rev_bounds)-2, -1, -1):
            if rev_bounds[i] is None and rev_bounds[i+1] is not None:
                rev_bounds[i] = brev(i, rev_bounds)

        # for i in range(len(bounds)):
        #     log.info(f"{i} {bounds[i]} {fwd_bounds[i]} {rev_bounds[i]}")

        # sanity check
        for i in range(len(bounds)):
            if bounds[i] is None:
                nb = int(fwd_bounds[i] is not None) + int(rev_bounds[i] is not None)
                if nb == 2:
                    if abs(fwd_bounds[i] - rev_bounds[i]) < 3:
                        bounds[i] = (fwd_bounds[i] + rev_bounds[i]) / 2;
                elif nb == 1:
                    bounds[i] = fwd_bounds[i] or rev_bounds[i]

        # now match again against bounds, not span
        # for i, b in enumerate(bounds):
        #     log(f"{i} {b}")
        for tl in self._non_header_tls():
            if 'row_ids' in tl._tmp:
                b = self._match_bounds(tl.bbox, bounds, 'y')
                if b is not None and b in tl._tmp['row_ids']:
                    tl._tmp['row_id'] = b
                else:
                    log.info(f"{tl} still undecided {tl.bbox}")

        if self.config.sections == 'empty-line':
            used_row_ids = {tl._tmp.get('row_id', -1) for tl in self._non_header_tls()}
            last_header_id = None
            for i, row in enumerate(row_h):
                if row._tmp['row_id'] not in used_row_ids:
                    row._tmp['section_header'] = True
                    last_header_id = i
                else:
                    row._tmp['section_row_id'] = last_header_id

    def _find_txt(self, seq, text):
        exact = [item for item in seq if item.text == text]
        if len(exact) == 1:
            return exact[0]
        inexact = [item for item in seq if text in item.text]
        if len(inexact) == 1:
            return inexact[0]
        raise ValueError(f"Failed to find text {text}")

    def _prep_fixed_cols(self):
        res = {}
        for col in self.fix.cols:
            cell = self._find_txt(self._header_col, col.row)
            row_id = cell._tmp['row_id']
            cids = []
            for span in col.spans:
                start = 0 if not cids else cids[-1][1]
                cids.append((start, start + span))
            res[row_id] = cids
        return res

    def _partition_cols_mid(self, heading_row=0):
        cells = list(self._row_certain())
        cells.sort(key=lambda tl: tl._tmp['row_id'])

        rows_midx = {}
        for row_id, row_cells in itertools.groupby(cells, lambda tl: tl._tmp['row_id']):
            row_cells = list(row_cells)
            sio = io.StringIO()
            sio.write(f"{row_id} -> ")
            for cell in row_cells:
                sio.write(f"{cell.text} ")
            log.info(sio.getvalue())
            # cells.sort(key=lambda tl: tl.bbox.midx)
            clusters = cluster(row_cells, lambda tl: tl.bbox.midx, eps=self._CLUSTER_COL_EPS)
            cluster_midx = [(cl[-1].bbox.midx + cl[0].bbox.midx) / 2 for cl in clusters]
            # log(f"{row_id} {cluster_midx}")
            rows_midx[row_id] = cluster_midx

        header_midx = rows_midx.get(heading_row)
        if header_midx is None:
            log.info(f"Can't find heading row={heading_row} cells")
            return
        ncols = len(header_midx)

        fixed_cols = self._prep_fixed_cols()

        # for row_id, midxs in rows_midx.items():
        for row_id, row_cells in itertools.groupby(cells, lambda tl: tl._tmp['row_id']):
            # drop stuff that is too early for first col
            row_cells = list(row_cells)
            orig_c = len(row_cells)
            row_cells = [cell for cell in row_cells
                         if cell.bbox.midx >= header_midx[0] - self._CLUSTER_COL_EPS]
            if len(row_cells) < orig_c:
                log.info(f"dropped early cells on row {row_id}")
            clusters = cluster(row_cells, lambda tl: tl.bbox.midx, eps=self._CLUSTER_COL_EPS)
            cluster_midx = [(cl[-1].bbox.midx + cl[0].bbox.midx) / 2 for cl in clusters]
            cids = None
            if row_id in fixed_cols:
                cids = fixed_cols[row_id]
            elif len(cluster_midx) == ncols:
                # log(f"{row_id} full")
                cids = [(i, i+1) for i in range(ncols)]
            elif len(cluster_midx) == 1:
                # log(f"{row_id} single")
                cids = [(0, ncols)]
            else:
                from_col = 0
                # log(f"{row_id} -->", end=' ')
                cids = []
                for midx in cluster_midx:
                    mid_col = 2 * from_col
                    while mid_col <= (ncols - 1) * 2:
                        if mid_col % 2 == 0:
                            rng = (header_midx[mid_col // 2],) * 2
                            coef = 1
                        else:
                            rng = (header_midx[mid_col // 2], header_midx[mid_col // 2 + 1])
                            coef = -1
                        rng = (rng[0] - coef * self._CLUSTER_COL_EPS,
                               rng[1] + coef * self._CLUSTER_COL_EPS)
                        if rng[0] <= midx and midx <= rng[1]:
                            break
                        mid_col += 1
                    if mid_col > (ncols - 1) * 2:
                        # log(f"{row_id} BAD midx {midx}", end=' ')
                        cids = None
                        break
                    to_col = mid_col - from_col
                    # log(f"[{from_col}-{to_col}]", end=' ')
                    cids.append((from_col, to_col+1))
                    from_col = to_col + 1

                if from_col != ncols:
                    log.info(f"FAIL !!!! END doesn't match")
                    cids = None
                else:
                    pass
            if cids:
                assert len(cids) == len(clusters)
                # if len(cids) != len(clusters):
                #     import pdb; pdb.set_trace()
                for cl_cells, colspan in zip(clusters, cids):
                    # TODO: Use joiner here
                    # import pdb; pdb.set_trace()
                    mcell = self._merge_cells(cl_cells, joiner=self._find_joiner(row_id))
                    mcell._tmp['col_ids'] = colspan
                    # print(f"{mcell}")

        # cells are merged now
        self._header_row = [cell for cell in self._row_certain()
                            if cell._tmp.get('row_id') == heading_row]
        log.info(f"{self._header_row}")
        self._header_row.sort(key=self._keyer('midx'))
        # self._header_row = self._header_row[1:]
        # for cell in self._header_row:
        #     cell._tmp['is_header'] = True

    def _find_joiner(self, row_id):
        joiner = ' '
        txt = self._header_col[row_id].text
        jd = [j for j in self.fix.joiner if j.row == txt]
        if len(jd) == 1:
            joiner = jd[0].seq
        return joiner

    def _partition_cols(self):
        cells = list(self._row_certain())
        cells.sort(key=lambda tl: tl._tmp['row_id'])
        for row_id, row_cells in itertools.groupby(cells, lambda tl: tl._tmp['row_id']):
            cells = list(row_cells)
            # log(f"ROWID {row_id}")
            # for cell in cells:
            #     log(f"--- {cell}")
            # cells.sort(key=lambda tl: tl.bbox.midx)
            clusters = cluster(cells, lambda tl: tl.bbox.midx, eps=self._CLUSTER_COL_EPS)
            cluster_midx = [(cl[-1].bbox.midx + cl[0].bbox.midx) / 2 for cl in clusters]
            cluster_edges = [None] * len(clusters)
            end = self.area.maxx
            for cid in reversed(range(len(clusters))):
                cluster_edges[cid] = cluster_midx[cid] * 2 - end
                end = cluster_edges[cid]
            # # sanity check
            # if abs(end - col_h[0]._span[0]) > self._SANITY_COL_EDGE_EPS:
            #     print(f"row {row_id} -- can't match column edge {abs(end - col_h[0]._span[0])}")
            #     continue
            cluster_cid = list(map(self._find_col_edge, cluster_edges))
            if any(cid is None for cid in cluster_cid):
                if len(clusters) == 1 and False: #self.opts.assume_one_all:
                    log.info(f"marking {row_id} {self._header_col[row_id].text} as applying to all cols")
                    cluster_cid[0] = 0 # starts from first one, ends with last one (default)
                else:
                    log.info(f"row {row_id} {self._header_col[row_id].text} -- can't match all column edges")
                    continue
            for i in range(len(clusters)):
                from_col = cluster_cid[i]
                to_col = cluster_cid[i+1] if i+1 < len(clusters) else len(self._header_row)
                for item in clusters[i]:
                    item._tmp['col_ids'] = (from_col, to_col)
                    # all_certain.append(item)

    def _get_sections(self, mid=None):
        for tl in self.textlines:
            if abs(tl.bbox.midx - mid) < 2:
                span = BBoxHelper.make_span((tl.bbox.miny, tl.bbox.maxy), 'y')
                inter = list(filter(lambda a: a != tl and a.bbox.intersects(span), self.textlines))
                if len(inter) == 0:
                    log.info(f"section {tl.text}")
                    tl._tmp['is_header'] = True
                    tl._tmp['is_section'] = True

    def _add_cells(self):
        for cell in self.fix.add_cell:
            row = self._find_txt(self.textlines, cell.row_text)
            col = self._find_txt(self.textlines, cell.col_text)
            bbox = BBoxHelper(col.bbox.minx, row.bbox.miny, col.bbox.maxx, row.bbox.maxy)
            self.textlines.append(TextLineHelper(cell.text, bbox, self.pdf.page(self.page_idx)))

    def _tabulate2(self):
        log.info("in here")
        self._add_cells()
        for tl in self.textlines:
            tl._tmp = {}
        if self.config.sections == 'table-center':
            self._get_sections(mid=(self.area.minx + self.area.maxx) / 2.0)
        row_h = self._get_row_header()
        # col_h = self._get_col_header()

        self._partition_rows()
        # self._dbg_show_cells(24)
        self._partition_cols_mid()
        # sys.exit(1)
        # self._partition_cols()

    def _dbg_show_cells(self, row_id=None):
        cells = list(self._non_header_tls())
        rcells = [c for c in cells if 'row_id' in c._tmp]
        rcells.sort(key=lambda c: (c._tmp['row_id'], c.bbox.midy))

        for rid, row_cells in itertools.groupby(rcells, lambda tl: tl._tmp['row_id']):
            if row_id is not None and row_id != rid:
                continue

            sio = io.StringIO()
            sio.write(f"{rid} --> ")
            row_cells = list(row_cells)
            row_cells.sort(key=self._keyer('midx'))
            for c in row_cells:
                sio.write(f"{c.text} {c.bbox} ")
            log.info(sio.getvalue())

        # last_row = None
        # for c in rcells:
        #     row = c._tmp['row_id']
        #     if row != last_row:
        #         last_row = row
        #         print(file=sys.stderr)
        #         if row_id is None or last_row == row_id:
        #             print(f"{row} --> ", file=sys.stderr, end=' ')

        #     if row_id is None or last_row == row_id:
        #         print(f"{c.text} {c.bbox}", end=' ', file=sys.stderr)

    # def _tabulate(self):
    #     # print('\n'.join(map(str, self.textlines)))


    #     # row_h = self._get_row_header()
    #     # col_h = self._get_col_header()


    #     row_h = self._find_first('minx')
    #     col_h = self._find_first('midy')

    #     # import pprint
    #     # pp = pprint.PrettyPrinter(indent=4)
    #     # pp.pprint(row_h)
    #     # pp.pprint(col_h)
    #     # print(f"ROW[0]: {row_h[0]}")
    #     # print(f"COL[0]: {col_h[0]}")

    #     for tl in self.textlines:
    #         tl._cell = None

    #     if row_h[0] == col_h[0]:
    #         row_h[0]._cell = (0, 0)

    #         row_h = row_h[1:]
    #         col_h = col_h[1:]

    #     for i, rh in enumerate(row_h, 1):
    #         rh._cell = (i, 0)
    #     for i, ch in enumerate(col_h, 1):
    #         ch._cell = (0, i)


    #     self._row_h = row_h
    #     self._col_h = col_h

    #     self._alloc_spans_nbr(row_h, 'y')
    #     self._alloc_spans(col_h, 'x', True)

    #     # for rh in row_h:
    #     #     print(f"{rh.text} {rh._span[1] - rh._span[0]} {rh._span}")
    #     # for ch in col_h:
    #     #     print(f"{ch.text} {ch._span[1] - ch._span[0]} {ch._span}")

    #     # inner = []
    #     cnts = [0, 0]
    #     disputed = defaultdict(int)
    #     row_certain = []
    #     for tl in self.textlines:
    #         if tl._cell is not None:
    #             continue
    #         # x = self._match_span(row_h, tl, 'y')
    #         # if x:
    #         #     tl._cell = (x, None)
    #         #     inner.append(tl)
    #         grp = self._match_span_multi(row_h, tl, 'y')
    #         if len(grp) == 1:
    #             tl._cell_row = grp[0]
    #             cnts[0] += 1
    #             row_certain.append(tl)
    #         else:
    #             print(f"-- dispute {tl} {tl._cell} {grp}")
    #             cnts[1] += 1
    #             tl._cell_rows = grp
    #             for gi in grp:
    #                 disputed[gi] += 1

    #     print(f"non-disputed: {cnts[0]} disputed {cnts[1]}")
    #     print(f"disputed rows: {disputed}")

    #     row_certain.sort(key=lambda tl: tl._cell_row)
    #     all_certain = []
    #     for row_id, row_cells in itertools.groupby(row_certain, lambda tl: tl._cell_row):
    #         cells = list(row_cells)
    #         # cells.sort(key=lambda tl: tl.bbox.midx)
    #         clusters = _cluster(cells, lambda tl: tl.bbox.midx, self._CLUSTER_COL_EPS)
    #         cluster_midx = [(cl[-1].bbox.midx + cl[0].bbox.midx) / 2 for cl in clusters]
    #         cluster_edges = [None] * len(clusters)
    #         end = self.area.maxx
    #         for cid in reversed(range(len(clusters))):
    #             cluster_edges[cid] = cluster_midx[cid] * 2 - end
    #             end = cluster_edges[cid]
    #         # # sanity check
    #         # if abs(end - col_h[0]._span[0]) > self._SANITY_COL_EDGE_EPS:
    #         #     print(f"row {row_id} -- can't match column edge {abs(end - col_h[0]._span[0])}")
    #         #     continue
    #         cluster_cid = list(map(self._find_col_edge, cluster_edges))
    #         if any(cid is None for cid in cluster_cid):
    #             print(f"row {row_id} -- can't match all column edges", file=sys.stderr)
    #             continue
    #         for i in range(len(clusters)):
    #             from_col = cluster_cid[i]
    #             to_col = cluster_cid[i+1] if i+1 < len(clusters) else len(self._col_h)+1
    #             for item in clusters[i]:
    #                 item._cell_col = from_col
    #                 item._cell_cs = to_col - from_col
    #                 all_certain.append(item)
    #         # y = self._match_span(col_h, tl, 'x')
    #         # self._set_row_col(tl, x, y)
    #         # print(f"{tl.text} {x} {y}")

    #     self._all_certain = all_certain

        # for cell in inner:
        #     if isinstance(cell._cell, tuple) and len(cell._cell) == 2 and cell._cell[1] is not None and len(cell._cell[1]) == 2:
        #         print(f"{cell.text} {cell._cell[0]} {cell._cell[1][0]}-{cell._cell[1][1]}")
        #     else:
        #         print(f"skip {cell.text}")

        # TODO(Iskren): Add sanity check
        # TODO(Iskren): Output in JSON


    # def _exp_cell(self, cell_spc):
    #     if cell_spc == None:
    #         return (None, None, None)
    #     elif isinstance(cell_spc, tuple) and len(cell_spc) == 2:
    #         assert isinstance(cell_spc[0], int)
    #         if cell_spc[1] is None:
    #             return (cell_spc[0], None, None)
    #         elif isinstance(cell_spc[1], int):
    #             return (cell_spc[0], cell_spc[1], cell_spc[1]+1)
    #         else:
    #             return (cell_spc[0], cell_spc[1][0], cell_spc[1][1])
    #     raise ValueError()

    def data_by_col(self):
        res = []
        for i, cell in enumerate(self._header_row):
            res.append(Column(i))

        # log(f"{len(self._header_row)} {self._header_row}")

        rc_certain = list(self._row_col_certain())
        # import pdb; pdb.set_trace()
        # in the same cell, we want text lines from top to bottom, so append will produce the right text
        rc_certain.sort(key=self._keyer('midy'))
        for cell in rc_certain:
            row = self._header_col[cell._tmp['row_id']]
            section = None
            if self.config.sections == 'empty-line':
                section_id = row._tmp['section_row_id']
                section = self._header_col[section_id].text if section_id is not None else None
            elif self.config.sections in ('first-col', 'table-center'):
                section_id = row._tmp['section_id']
                section = self._section_col[section_id].text if section_id >= 0 and section_id < len(self._section_col) else None
            # log(f"--- {cell._tmp['col_ids']}")
            cols = range(*cell._tmp['col_ids'])
            for col in cols:
                log.info(f"{section} {row.text} {cell.text}")
                res[col].add_attribute(section, row.text, cell.text)

        return res

    @classmethod
    def cols_to_rows(cls, cols):
        tbl = defaultdict(lambda: [None] * len(cols))
        for col_obj in cols:
            col_json = col_obj.to_json()
            col_id = col_json['idx']
            for section, s_attrs in col_json['props'].items():
                for row, val in s_attrs.items():
                    tbl[(section, row)][col_id] = val

        return [list(th) + list(tv) for (th, tv) in  tbl.items()]

    # def to_json(self):
    #     cells = []
    #     filtered = list(filter(lambda tl: tl._cell is not None and tl._cell[1] is not None, self.textlines))
    #     if len(filtered) != len(self.textlines):
    #         print(f"dropped {len(self.textlines) - len(filtered) }", file=sys.stderr)
    #     clusters = _cluster(filtered, lambda tl: tl._cell, eps=None)
    #     for cluster in clusters:
    #         # if cluster[0]._cell is None or cluster[0]._cell[1] is None:
    #         #     print(f"skipping {len(cluster)} cells", file=sys.stderr)
    #         #     continue
    #         cluster.sort(key=lambda tl: tl.bbox.miny)
    #         item = cluster[0]
    #         cells.append({
    #             'row': item._cell[0],
    #             'col': item._cell[1],
    #             'colspan': item._cell_cs if hasattr(item, '_cell_cs') else 1,
    #             'text': '\n'.join(tl.text for tl in cluster),
    #             'bbox': functools.reduce(lambda a, b: a.accomodating(b.bbox), cluster, BBoxHelper.EMPTY).to_json(),
    #         })

    #     return cells


class Config(BaseModel):
    file: str = Field(description="pdf filename to extract")
    area: NamedTuple("area", [('x', int), ('y', int), ('w', int), ('h', int)])
    page: int = Field(1, description="1-based page number to prase")
    out: str|None = Field(description="output file, defaults to stdout")
    class SectionType(str, Enum):
        # there are no sections
        NONE = 'none'
        # a row header without column data (i.e empty row) is a section
        EMPTY_LINE = 'empty-line'
        # there is a dedicated column (first) for section names
        FIRST_COL = 'first-col'
        # the section header is positioned at the exact y-center of the table
        TABLE_CENTER = 'table-center'
    sections: SectionType = Field(SectionType.NONE, description="section location in table")
    multiline_row_header: bool = False
    multiline_section_header: bool = False

    class FixData(BaseModel):
        class ColData(BaseModel):
            row: str
            spans: list[int]
            # texts: list[str] | None
        cols: list[ColData] = Field([], description="adjust ill-aligned cells")
        class AddCellData(BaseModel):
            """Add a cell with given text located at the intersection of cells given by row_text and col_text"""
            row_text: str
            col_text: str
            text: str
        add_cell: list[AddCellData] = Field([], description="add cell with given text")
        class JoinerData(BaseModel):
            row: str
            seq: str
        joiner: list[JoinerData] = Field([], description="how to merge multiline cells on a given row")
    fix: FixData = Field(FixData(), description="various adjustments to get things going")

    @classmethod
    def from_opts(cls, opts):
        if opts.action == 'extract-cfg':
            res = Config(**json.loads(opts.config))
        elif opts.action == 'extract':
            res = Config(
                file=opts.file,
                area=opts.area.split(','),
                page=opts.page,
                out=opts.out,
                section=opts.section,
                multiline_row_header=opts.multiline_row_header,
                multiline_section_header=opts.multiline_section_header,
                fix=json.loads(opts.fix or '{}'),
            )
        return res


def parse(args):
    parser = argparse.ArgumentParser("parse tabular data")
    actions = parser.add_subparsers(dest='action')

    extract = actions.add_parser('extract')
    extract.add_argument('-a', '--area', type=str, required=True,
                        help="x,y,w,h in pt of table")
    extract.add_argument('-p', '--page', type=int, default=1,
                        help="pdf page to use")
    extract.add_argument('-o', '--output', type=str,
                        help="output file, defaults to stdout")
    extract.add_argument('--verify', action='store_true',
                         help="Do not overwrite the output file, re-compute and show diff")
    extract.add_argument('--multiline-row-header', action='store_true',
                         help="Row header cells could be more than 1 line")
    extract.add_argument('--multiline-section-header', action='store_true',
                         help="Section header cells could be more than 1 line")
    extract.add_argument('--sections', choices=('empty-line', 'first-col', 'table-center', 'none'),
                         help="Where section headers positioned")
    # extract.add_argument('--assume-one-all', action='store_true',
    #                      help="Assume, that a single value in the columns applies to all columns (even if not centered correctly)")
    extract.add_argument('--fix', type=str, help="a json with data to help parse incorrect input")
    extract.add_argument('file', nargs=1, help="pdf file to parse")

    extract_cfg = actions.add_parser('extract-cfg')
    extract_cfg.add_argument('-c', '--config', type=str, required=True,
                             help="json-encoded configuration")
#     extract_yaml = actions.add_parser('extract-yaml')
#     extract_yaml.add_argument('-c', '--yaml', action='append',
#                               type=str, required=True,
#                               help="yaml config file")
#     extract_yaml.add_argument('-f', '--file', action='append',
#                               help='file to process')
#     extract_yaml.add_argument('-x', '--ignore-file', action='append',
#                               help='file to ignore')
#     extract_yaml.add_argument('--verify', action='store_true',
#                               help='don\'t override output files, just show diff')
#     extract_yaml.add_argument('-o', '--output',
#                               help='dir for extracted json files')
#     check_yaml = actions.add_parser('check-yaml')
#     check_yaml.add_argument('-c', '--yaml',
#                             type=str, required=True,
#                             help="yaml config file")
#     xml = actions.add_parser('xml')
#     xml.add_argument('file', nargs=1, help="pdf file to parse")
#     tl = actions.add_parser('textlines')
#     tl.add_argument('--page', default=0, type=int, help="page to show (from 0)")
#     tl.add_argument('file', nargs=1, help="pdf file to parse")

    return parser.parse_args(args)


def _collect_args(file, opts, cmdline):
    args = []
    args.append('extract')
    args.extend(('--page', opts['page'] - 1))
    args.extend(('--area', ','.join(map(str, opts['area']))))
    args.extend(('--sections', opts.get('sections', 'none')))
    for bool_arg in ['multiline-row-header', 'multiline-section-header', 'assume-one-all']:
        if opts.get(bool_arg):
            args.append('--' + bool_arg)
    if opts.get('fix'):
        args.extend(('--fix', json.dumps(opts['fix'])))
    if cmdline.verify:
        args.append('--verify')
    if cmdline.output:
        args.extend(('-o', file.parent / cmdline.output / (file.name + '.csv')))
    args.append(file)
    return list(map(str, args))

def show_diff(a, b, path=''):
    same = True
    if type(a) != type(b):
        log.info(f"diff at {path}: types differ {type(a)} {type(b)}")
        same = False
    elif isinstance(a, list):
        assert isinstance(b, list)
        min_len = min(len(a), len(b))
        if len(a) != len(b):
            log.info(f"diff at {path}: arr len differ {len(a)} {len(b)}")
        for i, (ia, ib) in enumerate(zip(a, b)):
            same = same and show_diff(ia, ib, path + f'[{i}]')
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        ka = set(a.keys())
        kb = set(b.keys())
        if ka != kb:
            msg = []
            if ka - kb:
                msg.append(f"a extra: {ka - kb}")
                same = False
            if kb - ka:
                msg.append(f"b extra: {kb - ka}")
                same = False
            log.info(f"diff at {path}: dict keys {','.join(msg)}")
        for ck in ka & kb:
            same = same and show_diff(a[ck], b[ck], path + f'.{ck}')
    else:
        # simple value
        if a != b:
            log.info(f"diff at {path}: atomic type diff {a} {b}")
            same = False
    return same


def main(args):
    opts = parse(args)
    if opts.action == 'extract' or opts.action == 'extract-cfg':
        config = Config.from_opts(opts)
        log.info(f"config: {json.dumps(config.dict(), indent=2)}")

        pdf = PDFDoc.from_file(config.file)
        tab = Tabulate(pdf, config)
        # tab._tabulate()
        # print(json.dumps(list(map(lambda x: x.to_json(), tab.data_by_col())), indent=2))
        tab._tabulate2()
        cols = tab.data_by_col()
        out = list(map(lambda x: x.to_json(), cols))
        if config.out:
            Path(config.out).parent.mkdir(parents=True, exist_ok=True)
            if config.out.endswith('.csv'):
                import csv
                with open(config.out, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    prev_line = [None]
                    writer.writerow(['section', 'property'] +
                                    [f'col:{i}' for i in range(len(cols))])
                    for line in Tabulate.cols_to_rows(cols):
                        pline = line
                        if line[0] == prev_line[0]:
                            # skip repeating section
                            pline = [None] + line[1:]
                        writer.writerow(pline)
                        prev_line = line
            else:
                output = config.out
                # if opts.verify and Path(output).exists():
                #     exp_out = json.loads(Path(output).read_text())
                #     outx = json.loads(json.dumps(out, indent=2))
                #     same = show_diff(outx, exp_out)
                #     if same:
                #         log(f"{opts.output} matches re-compute")
                #         output = None
                #     else:
                #         output += '.tmp'
                #         log(f"storing new data in {output}")
                if output:
                    with open(output, 'w') as f:
                        json.dump(out, f, indent=2)
        else:
            print(json.dumps(out, indent=2))
        # log(json.dumps(list(map(lambda x: x.to_json(), tab.data_by_col())), indent=2))
        # if opts.output:
        #     with open(opts.output, 'wb') as f:
        #         json.dump(tab.to_json(), f, indent=2)
        # else:
        #     json.dump(tab.to_json(), sys.stdout, indent=2)
    elif opts.action == 'extract-yaml':
        for yaml_file in opts.yaml:
            cfg = yaml.load(Path(yaml_file).read_bytes(), Loader=yaml.Loader)
            common = cfg.get('extractor', {})
            for item in cfg['pipelines']:
                if ((not opts.file or item['file'] in opts.file) and
                    (not opts.ignore_file or item['file'] not in opts.ignore_file)
                ):
                    for extr in item.get('extractors', [item.get('extractor')]):
                        if extr is None:
                            continue
                        # log("executing for {item['file']}")
                        item_opts = {}
                        item_opts.update(common)
                        item_opts.update(extr)
                        args = _collect_args(Path(yaml_file).parent / item['file'], item_opts, opts)
                        log.info(f"executing for {item['file']} {args}")
                        main(args)
    elif opts.action == 'check-yaml':
        yaml_path = Path(opts.yaml)
        cfg = yaml.load(yaml_path.read_bytes(), Loader=yaml.Loader)
        all_pdfs = set(file for file in yaml_path.parent.iterdir() if file.name.endswith('.pdf'))
        conf_pdfs = set()
        for item in cfg['pipelines']:
            pdf_path = yaml_path.parent / item['file']
            if not pdf_path.exists():
                log.info(f"{pdf_path} does not exist")
                continue
            conf_pdfs.add(pdf_path)

        if all_pdfs - conf_pdfs:
            for item in all_pdfs - conf_pdfs:
                log.info(f"non-configured pdf: {item}")
    elif opts.action == 'xml':
        xml = extract_pdfminer_xml(opts.file[0])
        print(xml)
    elif opts.action == 'textlines':
        pdf = PDFDoc.from_file(opts.file[0])
        page = pdf.page(opts.page)
        for tl in page._textlines:
            print(tl)

if __name__ == '__main__':
    main(sys.argv[1:])


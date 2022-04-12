"""process the corpus"""
# -*- encoding: utf-8 -*-
# @Author: RZH

import json
import re
from collections import defaultdict
from decimal import Decimal, getcontext
from os import listdir, path
from time import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        """no tqdm"""
        print('NOTICE: tqdm not installed, progress will not be shown')
        return it


class DataProcessor(object):
    """preprocess corpus data"""

    def __init__(self, corpus_path, prob_path):
        self.corpus = []
        self.prob_path = prob_path
        self.stat = defaultdict(lambda: defaultdict(int))
        self.prob = defaultdict(lambda: defaultdict(float))
        if path.isdir(corpus_path):
            self.corpus_path = list(map(lambda x: path.join(corpus_path, x), listdir(corpus_path)))
        elif path.isfile(corpus_path):
            self.corpus_path = [corpus_path]
        else:
            raise ValueError('corpus_path is not a file or directory')

    def _analyze(self):
        """count character and word pair occurrence"""
        raise NotImplementedError('_analyze method not implemented')

    def _calc_prob(self):
        """calculate probability"""
        raise NotImplementedError('_calc_prob method not implemented')

    def _read_corpus(self):
        """read corpus file"""
        for corpus_file in tqdm(self.corpus_path, desc='read corpus', unit='files'):
            with open(corpus_file, 'r', encoding='gbk') as f:
                while True:
                    try:
                        line = json.loads(f.readline().strip())
                    except json.JSONDecodeError:
                        break
                    self.corpus.clear()  # free memory
                    self.corpus += re.findall(r'[\u4e00-\u9fa5]+', line['html'].replace('原标题', ''))
                    self.corpus += re.findall(r'[\u4e00-\u9fa5]+', line['title'])
                    self._analyze()

    def process(self):
        """process corpus and build model"""
        print('start processing...')
        t = time()
        print('reading corpus...')
        self._read_corpus()
        print('doing statistic...')
        self._calc_prob()
        del self.stat  # free memory
        print('writing stat file...')
        with open(self.prob_path, 'w', encoding='utf-8') as f:
            json.dump(self.prob, f, ensure_ascii=False)
        print(f'preprocessing finished in {(time() - t):.2f} seconds')


class BiGramProcessor(DataProcessor):
    """data processor based on binary grammar"""

    def _analyze(self):
        for line in self.corpus:
            self.stat[''][''] += len(line)
            self.stat['^'][''] += 1
            self.stat[line[0]][''] += 1
            self.stat[line[0]]['^'] += 1  # first character as beginning
            self.stat['$'][line[-1]] += 1  # last character as ending
            for c, p in zip(line[1:], line):
                self.stat[c][p] += 1  # character `c` as successor of previous character `p`
                self.stat[c][''] += 1

    def _calc_prob(self):
        for c, s in tqdm(self.stat.items(), desc='calculate probability', unit='characters'):
            for p, o in s.items():
                self.prob[c][p] = float(Decimal(o) / self.stat[p][''])


class TriGramProcessor(DataProcessor):
    """data processor based on ternary grammar"""

    def _analyze(self):
        for line in self.corpus:
            self.stat[''][''] += len(line)
            self.stat['^'][''] += 1
            self.stat[line[0]][''] += 1
            self.stat[line[0]]['^'] += 1  # first character as beginning
            self.stat['$'][line[-1]] += 1  # last character as ending
            if len(line) > 1:
                self.stat[line[1]][line[0]] += 1  # second character as successor of first character
                self.stat[line[1]][''] += 1
            for c, p, q in zip(line[2:], line[1:], line):
                self.stat[c][q + p] += 1  # `c` as successor of previous two characters `qp`
                self.stat[q + p][''] += 1
                self.stat[c][p] += 1  # `c` as successor of previous character `p`
                self.stat[c][''] += 1

    def _calc_prob(self):
        for c, s in tqdm(
                list(filter(lambda it: len(it[0]) == 1, self.stat.items())),
                desc='calculate probability', unit='characters'):
            for p, o in s.items():
                if o == 1:
                    continue
                self.prob[c][p] = float(Decimal(o) / self.stat[p][''])


class QuadGramProcessor(DataProcessor):
    """data processor based on quadratic grammar"""

    def _analyze(self):
        for line in self.corpus:
            self.stat[''][''] += len(line)
            self.stat['^'][''] += 1
            self.stat[line[0]][''] += 1
            self.stat[line[0]]['^'] += 1  # first character as beginning
            self.stat['$'][line[-1]] += 1  # last character as ending
            if len(line) > 1:  # len = 2, ...
                self.stat[line[1]][line[0]] += 1  # second character as successor of first character
                self.stat[line[1]][''] += 1
                if len(line) > 2:  # len = 3, ...
                    # third character as successor of previous two characters
                    self.stat[line[2]][line[0] + line[1]] += 1
                    self.stat[line[0] + line[1]][''] += 1
                    # third character as successor of second character
                    self.stat[line[2]][line[1]] += 1
                    self.stat[line[2]][''] += 1
            for c, p, q, r in zip(line[3:], line[2:], line[1:], line):
                self.stat[c][r + q + p] += 1  # `c` as successor of previous three characters `rqp`
                self.stat[r + q + p][''] += 1
                self.stat[c][q + p] += 1  # `c` as successor of previous two characters `qp`
                self.stat[q + p][''] += 1
                self.stat[c][p] += 1  # `c` as successor of previous character `p`
                self.stat[c][''] += 1

    def _calc_prob(self):
        for c, s in tqdm(
                list(filter(lambda it: len(it[0]) == 1, self.stat.items())),
                desc='calculate probability', unit='characters'):
            for p, o in s.items():
                if o == 1:
                    continue
                self.prob[c][p] = float(Decimal(o) / self.stat[p][''])


def do_stat(corpus_path: str, stat_path: str, processor: str):
    """do statistic"""
    getcontext().prec = 5
    processors = {
        'bigram': BiGramProcessor,
        'trigram': TriGramProcessor,
        'quadgram': QuadGramProcessor,
    }
    processor = processors[processor](corpus_path, stat_path)
    processor.process()


if __name__ == '__main__':
    pass

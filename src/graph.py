# -*- encoding: utf-8 -*-
# @Author: RZH

import json
from math import log
from os import path
from time import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        """no tqdm"""
        print('NOTICE: tqdm not installed, progress will not be shown')
        return it

from validate import do_validate


class CharacterGraph(object):
    """character graph"""

    def __init__(self, params=None):
        # each node: (character, dist, parent)
        # each layer: [node1, node2, ...]
        self.params = params
        self.layers = [[('^', 0, None)]]
        with open(path.join(path.dirname(__file__), 'stat/mapping.json'), 'r', encoding='utf-8') \
                as f:
            self.mapping = json.load(f)
        self.prob = {}

    def _dist(self, char_prob: dict, p_node: tuple) -> float:
        """calculate distance between current character node and a parent node"""
        raise NotImplementedError('_dist not implemented')

    def _find_parent(self, char: str) -> tuple:
        """find parent of char that has the shortest distance"""
        dist = 0xffffffff
        parent = None
        char_prob = self.prob.get(char, {})
        for index, p_node in enumerate(self.layers[-1]):
            new_dist = self._dist(char_prob, p_node)
            if new_dist < dist:
                dist = new_dist
                parent = index
        return dist, parent

    def _add_layer(self, pinyin: str):
        """add layer to sentence graph"""
        layer = []
        chars = self.mapping[pinyin]
        for char in chars:
            layer.append((char, *self._find_parent(char)))
        self.layers.append(layer)

    def get_sentence(self, inputs: list) -> str:
        """get sentence from graph"""
        # init layers
        self.layers = [[('^', 0, None)]]
        # build layers
        for pinyin in inputs:
            self._add_layer(pinyin.strip().lower())
        self.layers.append([('$', *self._find_parent('$'))])
        # trace sentence
        sentence = ''
        p_index = self.layers[-1][0][2]
        for layer in self.layers[-2::-1]:
            sentence += layer[p_index][0]
            p_index = layer[p_index][2]
        return sentence[-2::-1]


class BiGramGraph(CharacterGraph):
    """graph for binary grammar"""

    def __init__(self, params=None):
        super().__init__(params if params else {'lambda': 0.99})
        with open(path.join(path.dirname(__file__), 'stat/bigram.json'), 'r', encoding='utf-8') \
                as f:
            self.prob = json.load(f)

    def _dist(self, char_prob: dict, p_node: tuple) -> float:
        """calculate distance between current character node and a parent node"""
        return p_node[1] - log(self.params['lambda'] * char_prob.get(p_node[0], 1e-7)
                               + (1 - self.params['lambda']) * char_prob.get('', 1e-4))


class TriGramGraph(CharacterGraph):
    """graph for ternary grammar"""

    def __init__(self, params=None):
        super().__init__(params if params else {'lambda': 0.39, 'mu': 0.6})
        with open(path.join(path.dirname(__file__), 'stat/trigram.json'), 'r', encoding='utf-8') \
                as f:
            self.prob = json.load(f)

    def _dist(self, char_prob: dict, p_node: tuple) -> float:
        """calculate distance between current character node and a parent node"""
        if len(self.layers) == 1:  # fallback to bigram
            return p_node[1] - log(self.params['lambda'] * char_prob.get(p_node[0], 1e-7)
                                   + (1 - self.params['lambda']) * char_prob.get('', 1e-4))
        pp_node = self.layers[-2][p_node[2]]
        return p_node[1] - log(self.params['mu'] * char_prob.get(pp_node[0] + p_node[0], 1e-6)
                               + self.params['lambda'] * char_prob.get(p_node[0], 1e-7)
                               + (1 - sum(self.params.values())) * char_prob.get('', 1e-4))


class QuadGramGraph(CharacterGraph):
    """graph for ternary grammar"""

    def __init__(self, params=None):
        super().__init__(params if params else {'lambda': 0.29, 'mu': 0.5, 'nu': 0.2})
        with open(path.join(path.dirname(__file__), 'stat/quadgram.json'), 'r', encoding='utf-8') \
                as f:
            self.prob = json.load(f)

    def _dist(self, char_prob: dict, p_node: tuple) -> float:
        """calculate distance between current character node and a parent node"""
        if len(self.layers) == 1:  # fallback to bigram
            return p_node[1] - log(self.params['lambda'] * char_prob.get(p_node[0], 1e-7)
                                   + (1 - self.params['lambda']) * char_prob.get('', 1e-4))
        pp_node = self.layers[-2][p_node[2]]
        if len(self.layers) == 2:  # fallback to trigram
            return p_node[1] - log(self.params['mu'] * char_prob.get(pp_node[0] + p_node[0], 1e-6)
                                   + self.params['lambda'] * char_prob.get(p_node[0], 1e-7)
                                   + (1 - sum(self.params.values())) * char_prob.get('', 1e-4))
        ppp_node = self.layers[-3][pp_node[2]]
        return p_node[1] - log(
            self.params['nu'] * char_prob.get(ppp_node[0] + pp_node[0] + p_node[0], 1e-5)
            + self.params['mu'] * char_prob.get(pp_node[0] + p_node[0], 1e-6)
            + self.params['lambda'] * char_prob.get(p_node[0], 1e-7)
            + (1 - sum(self.params.values())) * char_prob.get('', 1e-4))


def do_predict(pinyin_path: str, sentence_path: str, model: str):
    """predict sentence from pinyin"""
    print('loading model...')
    t = time()
    graphs = {
        'bigram': BiGramGraph,
        'trigram': TriGramGraph,
        'quadgram': QuadGramGraph
    }
    graph = graphs[model]()
    print(f'model loaded in {time() - t:.2f} seconds')
    print('start predicting...')
    t = time()
    with open(pinyin_path, 'r', encoding='utf-8') as pinyin, \
            open(sentence_path, 'w', encoding='utf-8') as sentence:
        for line in tqdm(pinyin, desc='predict', unit='sentence'):
            sentence.write(f'{graph.get_sentence(line.strip().split())}\n')
    print(f'prediction finished in {(time() - t):.2f} seconds')


def do_train(pinyin_path: str, sentence_path: str, model: str, params: list):
    """optimize parameters of model"""
    print('loading model...')
    t = time()
    graphs = {
        'bigram': BiGramGraph,
        'trigram': TriGramGraph,
        'quadgram': QuadGramGraph
    }
    graph = graphs[model]()
    print(f'model loaded in {time() - t:.2f} seconds')

    with open(pinyin_path, 'r', encoding='utf-8') as pinyin, \
            open(sentence_path, 'r', encoding='utf-8') as sentence:
        pinyin_lines = pinyin.readlines()
        sentence_lines = sentence.readlines()
    for param in params:
        graph.params = param
        output_lines = [graph.get_sentence(line.strip().split()) for line in pinyin_lines]
        print(f'==== Validate Result for {param} ====')
        do_validate(sentence_lines, output_lines)


if __name__ == '__main__':
    do_train('../data/test/input.txt', '../data/test/std_output.txt', 'trigram',
             [{'lambda': lam, 'mu': mu} for lam, mu in [
                 # (0.09, 0.9), (0.19, 0.8), (0.29, 0.7), (0.39, 0.6), (0.49, 0.5), (0.59, 0.4),
                 # (0.69, 0.3), (0.79, 0.2), (0.89, 0.1),
                 (0.35, 0.64), (0.37, 0.62), (0.39, 0.6), (0.41, 0.58), (0.43, 0.56), (0.45, 0.54),
             ]])
    pass

# -*- encoding: utf-8 -*-
# @Author: RZH

from typing import Union


def do_validate(std: Union[str, list], pred: Union[str, list]):
    """calculate precision of sentence prediction"""
    char_correct = 0
    char_total = 0
    sentence_correct = 0
    sentence_total = 0
    if isinstance(std, str):
        with open(std, 'r', encoding='utf-8') as std_file:
            std_lines = std_file.readlines()
    else:
        std_lines = std
    if isinstance(pred, str):
        with open(pred, 'r', encoding='utf-8') as pred_file:
            pred_lines = pred_file.readlines()
    else:
        pred_lines = pred
    assert len(std_lines) == len(pred_lines)

    for std_line, pred_line in zip(std_lines, pred_lines):
        std_line = std_line.strip()
        pred_line = pred_line.strip()
        char_total += len(std_line)
        sentence_total += 1
        char_correct += sum(map(lambda x: x[0] == x[1], zip(std_line, pred_line)))
        sentence_correct += std_line == pred_line
    char_precision = char_correct / char_total
    sentence_precision = sentence_correct / sentence_total
    print(f'char_precision: {char_precision:.4f}')
    print(f'sentence_precision: {sentence_precision:.4f}')
    return char_precision, sentence_precision


if __name__ == '__main__':
    pass

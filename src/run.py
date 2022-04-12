# -*- encoding: utf-8 -*-
# @Author: RZH

from argparse import ArgumentParser

from preprocess import do_stat
from graph import do_predict
from validate import do_validate


def main(task: str, model: str, input_path: str, output_path: str):
    """main entrance of the program"""
    if task == 'stat':
        do_stat(input_path, output_path, model)
    if task == 'predict':
        do_predict(input_path, output_path, model)
    if task == 'val':
        do_validate(input_path, output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='predict',
                        choices=['predict', 'stat', 'val'], help='task to run')
    parser.add_argument('-m', '--model', type=str, default='bigram',
                        choices=['bigram', 'trigram', 'quadgram'], help='model to apply')
    parser.add_argument('-i', '--input', type=str, required=True, help='input file')
    parser.add_argument('-o', '--output', type=str, required=True, help='output file')
    args = parser.parse_args()
    main(args.task, args.model, args.input, args.output)

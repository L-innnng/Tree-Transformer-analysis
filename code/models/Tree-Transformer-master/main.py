import argparse
from solver import Solver, Corpus
import torch
from utils import *
import pandas as pd


def get_raw_data(args):
      train_data = pd.DataFrame(Corpus().train_sens)
      test_data = pd.DataFrame( Corpus().test_sens)

      train_data.to_csv(args.train_path,header= False, index=None,doublequote =False, quoting=None)
      test_data.to_csv(args.test_path, header= False, index=None, doublequote =False, quoting=None)

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-no_cuda', action='store_true', default=True, help="Don't use GPUs.")
    parser.add_argument('-model_dir',default='code/models/Tree-Transformer-master/train_model',help='output model weight dir')
    parser.add_argument('-pp_dir',default='data/PP_ambiguity',help='output ppa ttachment ambiguity data dir')
    parser.add_argument('-gp_dir',default='data/garden_path',help='output garden path effect data dir')
    parser.add_argument('-seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_step', type=int, default=60000, help='sequence length')
    parser.add_argument('-data_dir',default='data_dir',help='data dir')
    parser.add_argument('-load',action='store_true',default=True, help='load pretrained model')
    parser.add_argument('-get_data', action='store_true', default=False, help='get raw penn treebank data for training and testing')
    parser.add_argument('-train', action='store_true',default=False,help='whether to train the model')
    parser.add_argument('-test', action='store_true',default=False,help='whether to test')
    parser.add_argument('-PP_ambiguity', action='store_true',default=False,help='whether to parse PP attachment ambiguity')
    parser.add_argument('-gp_MVRR', action='store_true',default=False,help='whether to calculate garden path effect on MVRR')
    parser.add_argument('-gp_overt_object', action='store_true',default=False,help='whether to calculate garden path effect on NP/Z overt_object')
    parser.add_argument('-gp_verb_transitivity', action='store_true',default=True,help='whether to calculate garden path effect on NP/Z verb_transitivity')
    parser.add_argument('-train_path',default='data/train.txt',help='training data path')
    parser.add_argument('-test_path',default='data/test.txt',help='testing data path')
    parser.add_argument('-PP_path',default='data/PP_ambiguity/pp_sentences.txt',help='pp attachment ambiguity data path')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
  
    args = parse()
    solver = Solver(args)

    if args.get_data:
      get_raw_data(args)
    elif args.train:
      solver.train()
    elif args.test:
      solver.test()
    elif args.PP_ambiguity:
      solver.PP_ambiguity()
    elif args.gp_MVRR:
      solver.GP_MVRR()
    elif args.gp_overt_object:
      solver.GP_overt_object()
    elif args.gp_verb_transitivity:
      solver.GP_verb_transitivity()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from parse import *
import random
from bert_optimizer import BertAdam
import nltk
from nltk.corpus import ptb
import numpy as np
import pandas as pd

from nltk import Tree
from functools import reduce

#from sklearn import metrics
import re
import pickle
import copy

from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import seaborn as sns


word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                    'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                    'WDT', 'WP', 'WP$', 'WRB', ',']#, ',', '.'

file_ids = ptb.fileids()
train_file_ids = []
valid_file_ids = []
test_file_ids = []
rest_file_ids = []
for id in file_ids:
    if 'WSJ/02/WSJ_0200.MRG' <= id <= 'WSJ/21/WSJ_2199.MRG':
        train_file_ids.append(id)
    if 'WSJ/22/WSJ_2200.MRG' <= id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(id)
    if 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
        test_file_ids.append(id)

class Corpus(object):
    def __init__(self):    
        self.train_sens, self.train_trees, self.train_nltktrees = self.tokenize(train_file_ids)
        self.valid_sens, self.valid_trees, self.valid_nltktress = self.tokenize(valid_file_ids)
        self.test_sens, self.test_trees, self.test_nltktrees = self.tokenize(test_file_ids)
        self.rest_sens, self.rest_trees, self.rest_nltktrees = self.tokenize(rest_file_ids)

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                words.append(w)
        return words

    def tokenize(self, file_ids):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    w = tree.leaves()[0].lower()
                    w = re.sub('[0-9]+', 'N', w)
                    return w
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        sens = []
        trees = []
        nltk_trees = []
        for id in file_ids:
            sentences = ptb.parsed_sents(id)

            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                sentence = " ".join(words)
                sens.append(sentence.strip('"'))
                trees.append(tree2list(sen_tree))
                nltk_trees.append(sen_tree)

        return sens, trees, nltk_trees

def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1

class Solver():
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.pp_dir = make_save_dir(args.pp_dir)
        self.gp_dir = make_save_dir(args.gp_dir)
        self.no_cuda = args.no_cuda
        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
        
        self.data_utils = data_utils(args)
        self.model = self._make_model(self.data_utils.vocab_size, 10)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        self.Step = 0
        self.loss = 0
        self.test_vecs = None
        self.test_masked_lm_input = []


    def _make_model(self, vocab_size, N=10, 
            d_model=512, d_ff=2048, h=8, dropout=0.1):
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
            group_attn = GroupAttention(d_model, no_cuda=self.no_cuda)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
            model = Encoder(EncoderLayer(d_model, c(attn), c(ff), group_attn, dropout), 
                    N, d_model, vocab_size, c(word_embed))
            
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.no_cuda:
                return model
            else:
                return model.cuda()

    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ttt = 1
                for  s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:',tt)

        data_yielder = self.data_utils.train_data_yielder()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

        
        total_loss = []
        start = time.time()
        total_step_time = 0.
        total_masked = 0.
        total_token = 0.

        for step in range(self.args.num_step):
            self.model.train()
            batch = data_yielder.__next__()
            
            step_start = time.time()
            out,break_probs = self.model.forward(batch['input'].long(), batch['input_mask'])
            
            loss = self.model.masked_lm_loss(out, batch['target_vec'].long())
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
            optim.step()

            total_loss.append(loss.detach().cpu().numpy())

            total_step_time += time.time() - step_start
            
            if step % 200 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Total Time: %f Step Time: %f" %
                        (step, np.mean(total_loss), elapsed, total_step_time))
                self.model.train()
                start = time.time()
                total_loss = []
                total_step_time = 0.


            if step % 1000 == 0:
                print('saving!!!!')
                
                model_name = 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))

    def test(self, threshold=0.8):
        print("test")
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
       
        self.model.eval()

        self.target_tree = Corpus().test_trees
        self.target_sen = Corpus().test_sens
        self.target_nltktrees = Corpus().test_nltktrees

        txts = get_test(self.args.test_path)
        vecs = [self.data_utils.text2id(txt, 60) for txt in txts]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs]
        self.test_vecs = cc(vecs, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts

        self.write_parse_tree()

    def write_parse_tree(self, threshold=0.8):
        batch_size = self.args.batch_size

        result_dir = os.path.join(self.model_dir, 'result/')
        make_save_dir(result_dir)
        f_b = open(os.path.join(result_dir,'brackets.txt'),'w')
        f_g = open(os.path.join(result_dir,'ground_brackets.txt'),'w')
        f1_list = []
        for b_id in range(int(len(self.test_txts)/batch_size)+1):

            out,attention,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])
            

            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(self.test_txts[b_id*batch_size+i].strip().split())

                bp = get_break_prob(break_probs[i])[:,1:length]
                model_out = build_tree(bp, 9, 0, length-1, threshold)
                if (0, length) in model_out:
                    model_out.remove((0, length))
                if length < 2:
                    model_out = set()
                f_b.write(json.dumps(list(model_out))+'\n')

                sentence = self.test_txts[b_id*batch_size+i]
                sen_tree = self.target_tree[b_id*batch_size+i]

                std_out, _ = get_brackets(sen_tree)
                f_g.write(json.dumps(list(std_out))+'\n')

                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                if len(model_out) == 0:
                    prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                f1_list.append(f1)
        
        f1_list = np.array(f1_list).reshape((-1,1))
        print('Mean F1:', np.mean(f1_list,axis=0))

        return np.mean(f1_list,axis=0)

    def PP_ambiguity(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()

        txts = get_test(self.args.PP_path)
        vecs = [self.data_utils.text2id(txt, 60) for txt in txts]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs]
        self.test_vecs = cc(vecs, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts
        self.labels = get_test(os.path.join(self.pp_dir, 'labels.txt'))
        batch_size = self.args.batch_size

        f_b = open(os.path.join(self.pp_dir,'tt_brackets.txt'),'w')
        f_gb = open(os.path.join(self.pp_dir,'ground_brackets.txt'),'w')

        i = 0

        N_tree = Tree.fromstring('(S(NP(N sentence[0]))(VP(V sentence[1])(NP(NP(Det sentence[2])(N sentence[3]))(PP(P sentence[4])(NP(Det sentence[5])(N sentence[6]))))))')
        Nstd_out, _ = get_brackets(N_tree)
        V_tree = Tree.fromstring('(S(NP(N sentence[0]))(VP(VP(V sentence[1])(NP(Det sentence[2])(N sentence[3])))(PP(P sentence[4])(NP(Det sentence[5])(N sentence[6])))))')
        Vstd_out, _ = get_brackets(V_tree)

        for b_id in range(int(len(self.test_txts)/batch_size)+1):

            x,out,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])
            
            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(self.test_txts[b_id*batch_size+i].strip().split())
                sentence = self.test_txts[b_id*batch_size+i].split()
                label = self.labels[b_id*batch_size+i]

                bp = get_break_prob(break_probs[i])[:,1:length]
                model_out = build_tree(bp, 9, 0, length-1, threshold)

                if (0, length) in model_out:
                    model_out.remove((0, length))
                if length < 2:
                    model_out = set()
                f_b.write(str(list(model_out))+'\n')

                if label == 'N' or label =='n':
                    std_out = Nstd_out
                elif label == 'V' or label =='v':
                    std_out = Vstd_out
                f_gb.write(str(list(std_out))+'\n')

    
    def GP_MVRR(self,threshold = 0.8):
        print("garden_path_MVRR")
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()

        txts = get_test(os.path.join(self.gp_dir , 'MVRR.txt'))
        vecs = [self.data_utils.text2id(txt, 60) for txt in txts]

        txts_formask = get_test(os.path.join(self.gp_dir , 'MVRR_mask.txt'))
        vecs_fromask = [self.data_utils.text2id(txt, 60) for txt in txts_formask]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs_fromask]

        self.test_vecs = cc(vecs_fromask, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts_formask
        batch_size = self.args.batch_size

        surprisal_list= []

        for b_id in range(int(len(self.test_txts)/batch_size)+1):
            x,attention,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])

            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(txts_formask[b_id*batch_size+i].strip().split())
                sentence = txts[b_id*batch_size+i]
                id = vecs[b_id*batch_size+i][length]
                prob = x[i]
                probs = torch.nn.functional.softmax(prob, dim = 1)
                surprisal = -torch.log2(probs)
                b = surprisal[length]
                surprisal_list.append(b[id].item())
        
        result11 = []
        result12 = []
        result2 = []
        result3 = []

        for i in range(0, len(surprisal_list), 4):
            result11.append(surprisal_list[i] - surprisal_list[i + 1])
            result12.append(surprisal_list[i+2] - surprisal_list[i + 3])
            result2.append(surprisal_list[i] - surprisal_list[i + 2])
            result3.append(surprisal_list[i] - surprisal_list[i + 1])

        ambiguity = np.array(result11)
        disambiguity = np.array(result12)

        am_mean = np.mean(ambiguity)
        dis_mean = np.mean(disambiguity)

        # Calculate the standard deviation
        am_std_dev = np.std(ambiguity, ddof=1)
        am_std = am_std_dev / np.sqrt(len(ambiguity))
        am_confidence_interval = 1.96 * np.array(am_std)

        dis_std_dev = np.std(disambiguity, ddof=1)
        dis_std = dis_std_dev / np.sqrt(len(disambiguity))
        dis_confidence_interval = 1.96 * np.array(dis_std)

        # Create lists for the plot
        materials = ['ambig', 'unambig']
        x_pos = np.arange(len(materials))
        means = [am_mean, dis_mean]
        error = [am_std, dis_std]
        confidence_interval = [am_confidence_interval,dis_confidence_interval]

        # Build the plot
        fig, ax = plt.subplots(figsize=(6, 5.1))
        ax.bar(x_pos, means, yerr=confidence_interval, capsize = 30,linewidth=50, zorder=100, color =['red','teal'], align='center', alpha=0.6, ecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(materials,fontsize=20)
        ax.set_ylabel('Garden path effect (bits)',fontsize=25)
        ax.set(ylim=(-0.6, 3.5))
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='both', linestyle='-', alpha=0.7, zorder=-100)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('mv_rr.png')
        plt.show()

    def GP_overt_object(self,threshold = 0.8):
        print("garden_path_overt_object")
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()

        txts = get_test(os.path.join(self.gp_dir , 'overt_object.txt'))
        vecs = [self.data_utils.text2id(txt, 60) for txt in txts]

        txts_formask = get_test(os.path.join(self.gp_dir , 'overt_object_mask.txt'))
        vecs_fromask = [self.data_utils.text2id(txt, 60) for txt in txts_formask]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs_fromask]

        self.test_vecs = cc(vecs_fromask, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts_formask
        batch_size = self.args.batch_size

        surprisal_list= []

        for b_id in range(int(len(self.test_txts)/batch_size)+1):
            x,attention,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])

            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(txts_formask[b_id*batch_size+i].strip().split())
                sentence = txts[b_id*batch_size+i]
                id = vecs[b_id*batch_size+i][length]
                prob = x[i]
                probs = torch.nn.functional.softmax(prob, dim = 1)
                surprisal = -torch.log2(probs)
                b = surprisal[length]
                surprisal_list.append(b[id].item())

        result11 = []
        result12 = []
        result2 = []
        result3 = []

        for i in range(0, len(surprisal_list), 4):
            result11.append(surprisal_list[i] - surprisal_list[i + 2])
            result12.append(surprisal_list[i+1] - surprisal_list[i + 3])
            result2.append(surprisal_list[i] - surprisal_list[i + 2])
            result3.append(surprisal_list[i] - surprisal_list[i + 1])

        ambiguity = np.array(result11)
        disambiguity = np.array(result12)

        am_mean = np.mean(ambiguity)
        dis_mean = np.mean(disambiguity)

        # Calculate the standard deviation
        am_std_dev = np.std(ambiguity, ddof=1)
        am_std = am_std_dev / np.sqrt(len(ambiguity))
        am_confidence_interval = 1.96 * np.array(am_std)

        dis_std_dev = np.std(disambiguity, ddof=1)
        dis_std = dis_std_dev / np.sqrt(len(disambiguity))
        dis_confidence_interval = 1.96 * np.array(dis_std)

        # Create lists for the plot
        materials = ['no-object', 'overt-object'] 
        x_pos = np.arange(len(materials))
        means = [am_mean, dis_mean]
        error = [am_std, dis_std]
        confidence_interval = [am_confidence_interval,dis_confidence_interval]

        # Build the plot
        fig, ax = plt.subplots(figsize=(6, 5.1))
        ax.bar(x_pos, means, yerr=confidence_interval, capsize = 30,linewidth=50, zorder=100, color =['red','teal'], align='center', alpha=0.6, ecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(materials,fontsize=20)
        ax.set_ylabel('Garden path effect (bits)',fontsize=25)
        ax.set(ylim=(-0.6, 3.5))
        ax.tick_params(axis='y', labelsize=20)
        #ax.yaxis.grid(True)
        ax.grid(axis='both', linestyle='-', alpha=0.7, zorder=-100)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('overt_object.png')
        plt.show()

    def GP_verb_transitivity(self,threshold = 0.8):
        print("garden_path_verb_transitivity")
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()

        txts = get_test(os.path.join(self.gp_dir , 'verb_transitivity.txt'))
        vecs = [self.data_utils.text2id(txt, 60) for txt in txts]

        txts_formask = get_test(os.path.join(self.gp_dir , 'verb_transitivity_mask.txt'))
        vecs_fromask = [self.data_utils.text2id(txt, 60) for txt in txts_formask]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs_fromask]

        self.test_vecs = cc(vecs_fromask, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts_formask
        batch_size = self.args.batch_size

        surprisal_list= []

        for b_id in range(int(len(self.test_txts)/batch_size)+1):
            x,attention,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])

            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(txts_formask[b_id*batch_size+i].strip().split())
                sentence = txts[b_id*batch_size+i]
                id = vecs[b_id*batch_size+i][length]
                prob = x[i]
                probs = torch.nn.functional.softmax(prob, dim = 1)
                surprisal = -torch.log2(probs)
                b = surprisal[length]
                surprisal_list.append(b[id].item())

        result11 = []
        result12 = []
        result2 = []
        result3 = []

        for i in range(0, len(surprisal_list), 4):
            result11.append(surprisal_list[i] - surprisal_list[i + 2])
            result12.append(surprisal_list[i+1] - surprisal_list[i + 3])
            result2.append(surprisal_list[i] - surprisal_list[i + 2])
            result3.append(surprisal_list[i] - surprisal_list[i + 1])

        ambiguity = np.array(result11)
        disambiguity = np.array(result12)

        am_mean = np.mean(ambiguity)
        dis_mean = np.mean(disambiguity)

        # Calculate the standard deviation
        am_std_dev = np.std(ambiguity, ddof=1)
        am_std = am_std_dev / np.sqrt(len(ambiguity))
        am_confidence_interval = 1.96 * np.array(am_std)

        dis_std_dev = np.std(disambiguity, ddof=1)
        dis_std = dis_std_dev / np.sqrt(len(disambiguity))
        dis_confidence_interval = 1.96 * np.array(dis_std)

        # Create lists for the plot
        materials = ['transitive', 'intransitive'] 
        x_pos = np.arange(len(materials))
        means = [am_mean, dis_mean]
        error = [am_std, dis_std]
        confidence_interval = [am_confidence_interval,dis_confidence_interval]

        # Build the plot
        fig, ax = plt.subplots(figsize=(6, 5.1))
        ax.bar(x_pos, means, yerr=confidence_interval, capsize = 30,linewidth=50, zorder=100, color =['red','teal'], align='center', alpha=0.6, ecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(materials,fontsize=20)
        ax.set_ylabel('Garden path effect (bits)',fontsize=25)
        ax.set(ylim=(-0.6, 3.5))
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='both', linestyle='-', alpha=0.7, zorder=-100)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('verb_transitivity.png')
        plt.show()


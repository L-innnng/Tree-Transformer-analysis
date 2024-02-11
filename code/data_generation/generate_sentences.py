import pandas as pd
import numpy as np
import nltk
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer 
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import torch
from nltk.corpus import wordnet as wn
import re
from transformers import pipeline

def verb_filter(text):
    ans = nltk.pos_tag(nltk.word_tokenize(text))
    val = ans[0][1]
    words = ['be','been','\'re','shown','seen','driven','done','\'m','gotten','gone','rung','taken','eaten']
    if(val == 'VBG' or val == 'POS' or (ans[0][0] in words) or len(re.findall(r'\b(\w+ing)\b',ans[0][0]))): 
        return False
    else:
        return True

def isn(text):
    ans = nltk.pos_tag(nltk.word_tokenize(text))
    val = ans[0][1]
    character = ['•','%','^','*','@','†','§','±','/',']','[','=','+','~','‡']
    if((val == 'NN' or val == 'NNS' or val == 'NNPS' or val == 'NNP') and (ans[0][0] not in character) and ans[0][0]!='please' and ans[0][0]!='it'):
        return True
    else:
        return False

def issubj(text):
    ans = nltk.pos_tag(nltk.word_tokenize(text))
    val = ans[0][1]
    words = ['he','she','it','they','we','you','I']
    if(ans[0][0] in words):
        return True
    else:
        return False

def find_subj(tokenizer, model, sentences, data):
    sents = []
    for index in range(len(sentences)):
        if isn(data.loc[index,'n1']) and isn(data.loc[index,'n2']) and verb_filter(data.loc[index,'v']):
            sentence = ' '.join(str(i) for i in sentences[index])
            pred = model(sentence)
            for s in pred:
                if issubj(s['token_str']) and s['token_str']!=data.loc[index,'n1'] and s['token_str']!=data.loc[index,'n2']:

                    data.loc[index,'subj'] = s['token_str']
                    s = data.loc[index,'subj':'c'].to_numpy()
                    sents.append(s)
                    break
        else:
            continue
                
    return sents

def ismodifier(text):
    ans = nltk.pos_tag(nltk.word_tokenize(text))
    val = ans[0][1]
    character = ['•','%','^','*','@','†','§','±','/',']','[','=','+','~','‡']
    pos = ['DT','NN','RBS','RBR','JJ','JJR','JJS', 'PRP$']
    if(val in pos and ans[0][0] not in character):
        return True
    else:
        return False

def find_nmodifier(tokenizer, model, sentences, data):
    sents = []
    for index in range(len(sentences)):
        sentence = ' '.join(str(i) for i in sentences[index])
        pred = model(sentence)
        for s in pred:
            if ismodifier(s['token_str']) and (s['token_str'] not in sentences[index]):
                data.loc[index,'MASK'] = s['token_str']
                s = (data.loc[index,'subj':'c']).to_numpy()
                sents.append(s)
                break
                
    return sents

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = pipeline('fill-mask', model='bert-base-uncased')

df = pd.read_csv('data/PP_ambiguity/PP_test', sep='\s+', header = None, names=['subj','v', 'n1', 'p','n2', 'c'])
df['subj']= '[MASK]'
df.drop_duplicates(inplace= True,ignore_index=True)
data = df.copy()
df.insert(loc=0, column='CLS', value='[CLS]')
df['DOT']= '.'
df['SEP']= '[SEP]'
df.pop('c')

sentences = df.loc[:,'CLS':'SEP'].to_numpy()
sents = find_subj(tokenizer, model, sentences, data)

data = pd.DataFrame(sents)
data.to_csv("data/PP_ambiguity/PP_subj.txt", header= False, index=None, sep=" ")

df = pd.read_csv('data/PP_ambiguity/PP_subj.txt', sep='\s+', header = None, names=['subj','v', 'n1', 'p','n2', 'c'])
df.insert(loc=0, column='CLS', value='[CLS]')
df.insert(loc=3, column='MASK', value='[MASK]')
df['DOT']= '.'
df['SEP']= '[SEP]'
mid = df['c']
df.pop('c')
df['c'] = mid
data = df.copy()
data.pop('SEP')

sentences = df.loc[:,'CLS':'SEP'].to_numpy()
sents = find_nmodifier(tokenizer, model, sentences, data)

data = pd.DataFrame(sents)
data.to_csv("data/PP_ambiguity/PP_m1.txt", header= False, index=None, sep=" ")

df = pd.read_csv('data/PP_ambiguity/PP_m1.txt', sep='\s+', header = None, names=['subj','v','m1','n1','p','n2','DOT','c'])
df.insert(loc=0, column='CLS', value='[CLS]')
df.insert(loc=6, column='MASK', value='[MASK]')
df['SEP']= '[SEP]'
mid = df['c']
df.pop('c')
df['c'] = mid
data = df.copy()
data.pop('SEP')

sentences = df.loc[:,'CLS':'SEP'].to_numpy()
sents = find_nmodifier(tokenizer, model, sentences, data)

data = pd.DataFrame(sents)
data.to_csv("data/PP_ambiguity/generated_PP_data.txt", header= False, index=None, sep=" ")

df = pd.read_csv('data/PP_ambiguity/generated_PP_data.txt', sep='\s+', header = None, names=['subj','v','m1','n1','p','m2','n2','DOT','c'])
data = df.loc[:,'subj':'n2']
data.to_csv("data/PP_ambiguity/pp_sentences.txt", header= False, index=None, sep=" ")
labels = df.loc[:,'c']
labels.to_csv("data/PP_ambiguity/labels.txt", header= False, index=None, sep=" ")

df = pd.read_csv('data/PP_ambiguity/generated_PP_data.txt', sep='\s+', header = None, names=['n0','v','m1','n1','p','m2','n2','d','l'])
df.pop('d')
trees=[]
sentences = df.to_numpy()
for sentence in sentences:
    n0 =nltk.pos_tag(nltk.word_tokenize(sentence[0]))[0][1]
    v= nltk.pos_tag(nltk.word_tokenize(sentence[1]))[0][1]
    if v =='NN':
        v='VB'
    elif v =='NNS':
        v= 'VBS'
    m1 = nltk.pos_tag(nltk.word_tokenize(sentence[2]))[0][1]
    n1 = nltk.pos_tag(nltk.word_tokenize(sentence[3]))[0][1]
    p = nltk.pos_tag(nltk.word_tokenize(sentence[4]))[0][1]
    m2 = nltk.pos_tag(nltk.word_tokenize(sentence[5]))[0][1]
    n2 = nltk.pos_tag(nltk.word_tokenize(sentence[6]))[0][1]
    if sentence[7] == 'V':
        tree = '(S(NP('+ n0 +' '+ sentence[0]+ '))(VP(VP('+v +' '+sentence[1]+ ')(NP('+ m1+' '+sentence[2]+')('+n1+' '+sentence[3]+')))(PP('+p+' '+sentence[4]+')(NP('+m2+' '+ sentence[5]+')('+n2+' ' +sentence[6]+')))))'
    else:
        tree = '(S(NP('+ n0 +' '+ sentence[0]+ '))(VP('+v +' '+sentence[1]+ ')(NP(NP('+ m1+' '+sentence[2]+')('+n1+' '+sentence[3]+'))(PP('+p+' '+sentence[4]+')(NP('+m2+' '+ sentence[5]+')('+n2+' ' +sentence[6]+'))))))'
    trees.append(tree)

trees = pd.DataFrame(trees)
trees.to_csv("data/PP_ambiguity/ground_trees.txt", header= False, index=None, quoting=None)



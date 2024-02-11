import pandas as pd
import numpy as np
import json
import warnings
from IPython.display import display_html
from collections import Counter
import nltk
from nltk import Tree
import matplotlib.pyplot as plt
from prettytable import PrettyTable

warnings.filterwarnings('ignore')

vbrackets = ['[(1, 7), (2, 4), (1, 4), (4, 7), (5, 7)]','[(4, 7), (1, 7), (2, 4), (1, 4)]','[(4, 7), (1, 7), (2, 4), (5, 7)]','[(2, 4), (1, 4)]']
nbrackets = ['[(1, 7), (2, 4), (2, 7), (4, 7), (5, 7)]','[(4, 7), (1, 7), (2, 4), (2, 7)]','[(2, 7), (1, 7), (4, 7), (5, 7)]','[(1, 7), (4, 7), (2, 7)]'] 
right_brackets = ['[(1, 7), (2, 4), (1, 4), (4, 7), (5, 7)]','[(4, 7), (1, 7), (2, 4), (1, 4)]','[(4, 7), (1, 7), (2, 4), (5, 7)]','[(2, 4), (1, 4)]','[(1, 7), (2, 4), (2, 7), (4, 7), (5, 7)]','[(4, 7), (1, 7), (2, 4), (2, 7)]','[(2, 7), (1, 7), (4, 7), (5, 7)]','[(1, 7), (4, 7), (2, 7)]']

pp = pd.read_csv('data/PP_ambiguity/pp_sentences.txt', sep='\s+', header = None, names=['subj','v','m1','n1','p','m2','n2'])
labels = pd.read_csv('data/PP_ambiguity/labels.txt', header = None, names= ['label'])
brackets = pd.read_csv('data/PP_ambiguity/BiLSTM_brackets.txt', dtype={'uid':str}, sep='\[]', header = None, names=['bracket'])
data = pd.concat([pp,labels,brackets], axis=1)

num_N = len(data[data['label'] == 'N'])
num_V = len(data[data['label'] == 'V'])

VT = data[(data['bracket'].isin(vbrackets)) & (data['label'] == 'V')]
VF = data[(data['bracket'].isin(nbrackets)) & (data['label'] == 'V')]
NT = data[(data['bracket'].isin(nbrackets)) & (data['label'] == 'N')]
NF = data[(data['bracket'].isin(vbrackets)) & (data['label'] == 'N')]
VW = data[(~data['bracket'].isin(right_brackets)) & (data['label'] == 'V')]
NW = data[(~data['bracket'].isin(right_brackets)) & (data['label'] == 'N')]

top_num = 20

def counters(c,word,data):
    num = Counter(data[c])[word]
    VT_word = Counter(VT[c])[word]
    VF_word = Counter(VF[c])[word]
    NT_word = Counter(NT[c])[word]
    NF_word =  Counter(NF[c])[word]
    VW_word =  Counter(VW[c])[word]
    NW_word =  Counter(NW[c])[word]

    table_data = [(VT_word, VF_word, VW_word, VT_word+VF_word), (NF_word, NT_word,NW_word, NT_word+NF_word), ( VT_word+NF_word, VF_word+NT_word,VW_word+NW_word, num)]
    df = pd.DataFrame(table_data, columns =["V_pre", "N_pre", 'Neither', "Sum"], index =["V_label", "N_label", "Sum"])
    df = df.style.set_table_attributes("style='display:inline'").set_caption("Table for" +' \''+ word +'\'')
    return df

def proportions(c,word,data):
    V_num = len(VT[VT[c]==word]) + len(NF[NF[c]==word])
    N_num = len(NT[NT[c]==word]) + len(VF[VF[c]==word])
    Neither_num = len(VW[VW[c]==word]) + len(NW[NW[c]==word])
    acc = (len(VT[VT[c]==word])+ len(NT[NT[c]==word]))/(V_num+N_num+Neither_num+0.00001)

    return V_num,N_num,acc,Neither_num

def fre_model(mark):
    c=mark
    p_counter = Counter(data[c]) 
    top_p = p_counter.most_common(top_num)

    prepositions = []
    V_proportions = []
    N_proportions = []
    Neither_proportions = []
    accs = []
    for i in range(top_num):
        V_num, N_num, acc, Neither_num= proportions(c,top_p[i][0],data)
        V_proportions.append(V_num/top_p[i][1])
        N_proportions.append(N_num/top_p[i][1])
        Neither_proportions.append(Neither_num/top_p[i][1])
        prepositions.append(top_p[i][0])
        accs.append(acc)
        
    species = prepositions
    weight_counts = {
        "Noun Attachement": N_proportions,
        "Verb Attachement": V_proportions,
        "Neither": Neither_proportions,
    }

    width = 0.5
    fig, ax = plt.subplots(figsize=(16, 4))
    bottom = np.zeros(len(prepositions))

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    plt.axhline(y=0.5, color='r', linestyle='--')
    #ax.set_title("Noun vs verb attachment proportions for frequent prepositions by model")
    ax.legend(loc="upper right")

    plt.savefig(mark+'_portion_model.png')
    plt.show()

def matrics(data,caption,precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy):
    table_data = [(len(VT), len(VF), len(VW), num_V),(len(NF), len(NT), len(NW), num_N), (len(VT)+len(NF), len(NT)+len(VF), len(VW)+len(NW), len(data))]
    df = pd.DataFrame(table_data, columns=["V_pre", "N_pre", "Neither", "Sum"],  index=["V_label", "N_label", "Sum"])
    df = df.style.set_table_attributes("style='display:inline'").set_caption(caption)
    display_html(df._repr_html_(), raw=True)

    precision = len(VT)/(len(VT)+len(NF))
    recall = len(VT)/num_V
    F1 = 2*precision*recall/(precision+recall)

    precision_v.append(precision)
    recall_v.append(recall)
    f1_score_v.append(F1)

    precision = len(NT)/(len(NT)+len(VF))
    recall = len(NT)/num_N
    F1 = 2*precision*recall/(precision+recall)

    precision_n.append(precision)
    recall_n.append(recall)
    f1_score_n.append(F1)

    accuracy.append((len(VT)+len(NT))/len(data))

    return precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy


# 1. Parsing result
table_data = [("V_label", len(VT), len(VF), len(VW), num_V), ("N_label", len(NF),len(NT), len(NW), num_N), ("Sum",len(VT)+len(NF) , len(NT)+len(VF), len(VW)+len(NW), len(data))]
table = PrettyTable()
table.field_names = [" ", "V_pre", "N_pre", "Neither", "Sum"]
for row in table_data:
    table.add_row(row)
print(table)

# 2. accuracy
accuracy = (len(VT)+len(NT))/len(data)
print('accuracy:',accuracy)

# 3. Noun vs verb attachment proportions for frequent prepositions/verbs from the BiLSTM.
fre_model('p')
fre_model('v')

# 4. Be/have confusion matrix

precision_v = []
recall_v = []
f1_score_v = []

precision_n = []
recall_n = []
f1_score_n = []

accuracy = []
bverbs = ['be','been','is','was','are','were','have','has','had'] 
bverb_data = data[data['v'].isin(bverbs)]
nverb_data = data[-data['v'].isin(bverbs)]

precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy = matrics(data,"Table for all data",precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy)
precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy = matrics(bverb_data,'Table for to-be and helping verbs',precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy)
precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy = matrics(nverb_data,'Table for notional verbs',precision_v,recall_v,f1_score_v,precision_n,recall_n,f1_score_n,accuracy)

fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex = True, sharey=True, layout='constrained')

species = ("all verbs", "be&have verbs", "notional verbs")
penguin_means = {
    'Precision': [ round(elem, 2) for elem in precision_n ],
    'Recall': [ round(elem, 2) for elem in recall_n ] ,
    'F1 score': [ round(elem, 2) for elem in f1_score_n ],
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = axs[0].bar(x + offset, measurement, width, label=attribute)
    axs[0].bar_label(rects, padding=3)
    multiplier += 1

axs[0].set_title('Attachment Decision is N')
axs[0].set_xticks(x + width, species)
axs[0].set_ylim(0, 1)

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

penguin_means = {
    'Precision': [ round(elem, 2) for elem in precision_v],
    'Recall': [ round(elem, 2) for elem in recall_v],
    'F1 score': [ round(elem, 2) for elem in f1_score_v],
}

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = axs[1].bar(x + offset, measurement, width, label=attribute)
    axs[1].bar_label(rects, padding=3)
    multiplier += 1

axs[1].set_title('Attachment Decision is V')
axs[1].set_xticks(x + width, species) 
axs[1].legend(loc='upper right',bbox_to_anchor=(1.35, 0.7),ncols=1)
axs[1].set_ylim(0, 1)

plt.savefig('verb_matric.png')
plt.show()
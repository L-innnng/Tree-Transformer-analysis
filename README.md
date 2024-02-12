# Tree Transformer's Disambiguation Ability of Prepositional Phrase Attachment and Garden Path Effects
This repository contains the source code from the papers ["Tree Transformer: Integrating Tree Structures into Self-Attention"](https://arxiv.org/abs/1909.06639) and ["Constituent Parsing as Sequence Labeling"](https://arxiv.org/abs/1810.08994). 

# Experiments on the Tree Transformer model
With `code/models/Tree-Transformer-master/main.py`, we can easily run the following tasks:

- `-get_data` : get raw penn treebank data for training and testing
- `-train`: train the  Tree Transformer model
- `-test`: test the  Tree Transformer model
- `-PP_ambiguity`: parse PP attachment ambiguity
- `-gp_MVRR`: calculate garden path effect on MVRR
- `-gp_overt_object`: calculate garden path effect on NP/Z overt_object
- `-gp_verb_transitivity`: calculate garden path effect on NP/Z verb_transitivity

With code/parsing_analysis/Tree_Transformer_parsing.py, we can get the results of parsing analysis for the Tree Transformer model.

# Experiments on the pretrained BiLSTM model

**Additional resources** You also might need to download the [pretrained models](http://grupolys.org/software/tree2labels-emnlp2018-resources/models-EMNLP2018.zip) and/or the [pretrained word embeddings](http://grupolys.org/software/tree2labels-emnlp2018-resources/embeddings-EMNLP2018.zip) used in this work to the folder tree2labels-master.

- First, using `code/models/tree2labels-master/dataset.py` to transform parenthesized trees into a sequence of labels.
- Second, using `code/models/tree2labels-master/main.py` to parse PP attachment ambiguity.
- Third,using `code/parsing_analysis/BiLSTM_parsing.py` to get the results of parsing analysis for the pretrained BiLSTM model.

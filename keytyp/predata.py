import random
import numpy as np
import torch
import json
import csv

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1234)

# TODO:テキストデータから読み出す
fname = ""

with open(fname) as f:
  for cols in csv.reader(f, delimiter='\t'):
    print(cols)
  with open(f'random_{fname}.csv') as f2:
    writer = csv.writer(f2, delimiter='\t')
    
    for data in json.load(f):
    # if fname.endswith(".csv"):
    #   pass      
    # elif fname.endswith(".tsv"):
    #   pass
    # elif fname.endswith(".jsonl"):
    #   pass


typ = "120 p 30 o 90 u 10 n 30 m 110 t 20 s 80 l 0 c 20 z" #事前学習用のテキストデータを入れる

typ_ = typ.split(' ')           # 半角区切りでリストに格納する
mask = len(typ_) * 0.40   #BERT: masked_lm_prob 4割にマスクをかける
typ_idx = [idx for idx in range(len(typ_))]

typ_mask = []
for index in typ_idx:
#   covered_indexes.add(index)
#   masked_token = None
  if random.random() < 1 - 0.8:       # 80% of the time, replace with [MASK]
    masked_token = "[MASK]"
    typ_mask.append(masked_token)
  else: # 10% of the time, keep original
    if random.random() < 0.5: 
      masked_token = typ_[index]
      typ_mask.append(masked_token)
    else:  # 10% of the time, replace with random word
      masked_token = typ_[random.randint(0, len(typ_idx) - 1)]
      typ_mask.append(masked_token)

# TODO:マスクしたデータをテキスト形式で書き出す
# SRC=マスクデータ、TGT=テキストデータ
import random
import numpy as np
import json
import csv
import pandas as pd
import re

# 事前学習用テキストデータから読み出す --> SRC, TGTのペアになるtsvファイルを出力
readfile = "/content/conala-mini.txt" 
writefile = "random_mask40.tsv"

def get_token(text): #conalaのBleuからtokenizerを拝借
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]
    return tokens

def mask(typ_, typ_idx):
  typ_mask = []
  for index in typ_idx:
    if random.random() < 0.4:       # Mask rate
      if random.random() < 0.8:     # 40% of the time, replace with [MASK]
        masked_token = "[MASK]"
        typ_mask.append(masked_token)
      elif random.random() > 0.8 and random.random() < 0.9: # 10% of the time, replace with random word
        masked_token = typ_[random.randint(0, len(typ_idx) - 1)]
        typ_mask.append(masked_token)
      else: # 10% of the time, keep original
        typ_mask.append(typ_[index])
    else:
      typ_mask.append(typ_[index])
  typ_mask = "".join(typ_mask)
  return typ_mask #SRC

def main():
  with open(readfile) as f: #読み込み用ファイル
    with open(writefile, 'w') as f2:
      writer = csv.writer(f2, delimiter='\t')
      for line in f:
        token_ = get_token(line)
        token_idx = [idx for idx in range(len(token_))] 
        token_mask = mask(token_, token_idx)
        
        src_tgt = [token_mask, line]
        print(src_tgt)
        writer.writerow(src_tgt)
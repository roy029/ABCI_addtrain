import random
import numpy as np
import torch
import json
import csv
import pandas as pd
import re

# 事前学習用テキストデータから読み出す --> SRC, TGTのペアになるtsvファイルを出力
readfile = "/content/conala-mini.txt"
writefile = "random_identifer.tsv"

def get_token(text):
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]
    return tokens

def identifer(typ_, typ_idx):
  typ_it = []
  for index in typ_idx:
    tf = typ_[index].isidentifier()
    typ_it.append(int(tf))
  return typ_it #SRC, TGT

with open(readfile) as f: #読み込み用ファイル
  with open(writefile, 'w') as f2:
    writer = csv.writer(f2, delimiter='\t')
    for line in f:
      # print(line)
      token_ = get_token(line)
      token_idx = [idx for idx in range(len(token_))]
      token_it = identifer(token_, token_idx)
      src_tgt = [line , token_it]
      print(src_tgt)
      writer.writerow(src_tgt)
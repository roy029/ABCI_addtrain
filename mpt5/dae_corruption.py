import random
import numpy as np
import torch
import json
import csv
import pandas as pd
import re

# 事前学習用テキストデータから読み出す --> SRC, TGTのペアになるtsvファイルを出力
readfile = "/content/conala-mined.txt"
writefile = "/content/random_dae40.tsv"

def get_token(text):
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]
    return tokens

def DAE(typ_, typ_idx):
  typ_cor = []
  for index in typ_idx:
    if random.random() < 0.40:       # 破壊率
      if random.random() < 1/4:     # 25%で除去(ok)
        pass
      elif random.random() > 1/4 and random.random() < 2/4: # 25%でシャッフル
        if index != len(typ_idx) and index != len(typ_idx)-1 and index != len(typ_idx)-2: #対象のindexと後ろ2つをシャッフル
          shuffle_lst = [index, index+1, index+2]
          tmp = random.sample(shuffle_lst, 3) #シャッフル
          if tmp == shuffle_lst:  #高確率でシャッフルに失敗するので
            tmp = random.sample(shuffle_lst, 3)
          typ_cor.append(typ_[tmp[0]])
          typ_cor.append(typ_[tmp[1]])
          typ_cor.append(typ_[tmp[2]])
          index += 2
        elif index != len(typ_idx) and index != len(typ_idx)-1: #対象のindexとすぐ後ろをシャッフル
          typ_cor.append(typ_[index+1])
          typ_cor.append(typ_[index])
          index += 1
        else: #シャッフルを諦める
          pass
      elif random.random() > 2/4 and random.random() < 3/4: # 25%でマスク(ok)
        masked_token = "[MASK]"
        typ_cor.append(masked_token)
      else:                                                  # 25%でニュータイプ
        masked_token = typ_[index]
        typ_cor.append(masked_token)
    else:
      typ_cor.append(typ_[index])
  typ_cor = "".join(typ_cor)
  return typ_cor


with open(readfile) as f: #読み込み用ファイル
  for line in f:
    # print(line)
    token_ = get_token(line)
    token_idx = [idx for idx in range(len(token_))] 
    token_cor = DAE(token_, token_idx)
    src_tgt = [token_cor, line]
    print(src_tgt)
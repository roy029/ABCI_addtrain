import random
import numpy as np
import json
import csv
import pandas as pd
import re

def get_token(text): #conalaのBleuからtokenizerを拝借
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]
    return tokens

def mask(s:str, ratio):
  token = get_token(s)
  token_idx = [idx for idx in range(len(token))] 
  buffer = []
  
  for index in token_idx:
    if random.random() < ratio:       # Mask rate
      if random.random() < 0.8:     # 40% of the time, replace with [MASK]
        masked_token = "[MASK]"
        if index == 0:
          buffer.append(masked_token)
        else:
          if masked_token != buffer[-1]: #ひとつ前がMASKではなかったらMASKを追加
            buffer.append(masked_token)
          else:
            pass
      elif random.random() > 0.8 and random.random() < 0.9: # 10% of the time, replace with random word
        masked_token = token[random.randint(0, len(token_idx) - 1)]
        buffer.append(masked_token)
      else:      # 10% of the time, keep original
        buffer.append(token[index])
    else:
     buffer.append(token[index])
  buffer = "".join(buffer)
  return buffer

# 事前学習用テキストデータから読み出す
readfile = "/content/conala-mined.txt"
writefile = "/content/random_mask40.txt"

with open(readfile) as f: #読み込み用ファイル
  with open(writefile, 'w') as f2:
    for line in f:
      token_mask = mask(line, 0.4)
      f2.write(token_mask + '\n')
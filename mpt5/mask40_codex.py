import random
import re
import json
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO
import csv
import glob

def get_token(text): #conala_Bleu from tokenizer
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('\n', ' <nl> ')
    text = text.replace('    ', ' <tab> ')
    text = text.replace('\t', ' <tab> ')
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]
    return tokens

def py_token(text):
  lst = []
  tokens = tokenize(BytesIO(text.encode('utf-8')).readline)
  for token in tokens:
    lst.append(token.string)
  return lst[1:-2]

def mask(token, ratio):
  token_idx = [idx for idx in range(len(token))] 
  buffer = []
  num = 0

  for index in token_idx:
    if random.random() < ratio:       # Mask rate
      if random.random() < 0.8:     # 40% of the time, replace with [MASK]
        masked_token = f'<extra_id_{num}>'
        if index == 0 and num < 100:
          buffer.append(masked_token)
          num += 1
        else:
          if "<extra_id" not in buffer[-1] and num < 100: #Add MASK if the previous one was not MASK.
            buffer.append(masked_token)
            num += 1
          else:
            pass
      elif random.random() > 0.8 and random.random() < 0.9: # 10% of the time, replace with random word
        input_token = token[random.randint(0, len(token_idx) - 1)]
        buffer.append(input_token)
        # num += 1
      else:      # 10% of the time, keep original
        buffer.append(token[index])
    else:
     buffer.append(token[index])
  buffer = "".join(buffer)
  return buffer

# 書き出しファイル
writefile = "/home/acd13734km/pydata/python/codex_mask40.tsv"

json_files = glob.glob("/home/acd13734km/pydata/python/final/jsonl/*/*.jsonl")
for file in json_files:
  with open(file) as f: #読み込み用ファイル
    with open(writefile, 'w') as f2:
      writer = csv.writer(f2, delimiter="\t")
      for line in f.readlines():
        data = json.loads(line)
        token = get_token(data["code"])
        token_mask = mask(token, 0.40)
        token_code = "".join(token)
        writer.writerow([token_mask, token_code])

# with open(readfile) as f: #読み込み用ファイル
#   with open(writefile, 'w') as f2:
#     writer = csv.writer(f2, delimiter="\t")
#     for line in f.readlines():
#       data = json.loads(line)
#       token = get_token(data["code"])
#       token_mask = mask(token, 0.15)
#       token_code = "".join(token)
#       writer.writerow([token_mask, token_code])


# with open(readfile) as f: #For text data
#   # with open(writefile, 'w') as f2:
#   for line in f:
#     token_mask = mask(line, 0.4)
#     print(token_mask)
#       # f2.write(token_mask + '\n')

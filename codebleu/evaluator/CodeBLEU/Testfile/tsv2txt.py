# 使い方
# python3 tsv2txt.py result_test.tsv(モデルからの生成結果)

# 実行すると、CodeBLEUに必要な、refs.txtとhypp.txtができます。
# CodeBLEUを実行するときに、--refs, --hypの後に与えるテキストファイル

import argparse
import json
import csv

parser = argparse.ArgumentParser(description='resultのtsvからテキストファイルを作るコード(仮)')
parser.add_argument('arg1', help='生成したresult.tsv') 
# parser.add_argument('--arg2', help='hypothesisファイルの名前') #まだ
# parser.add_argument('--arg3', help='referenceファイルの名前') #まだ

args = parser.parse_args() 

with open(args.arg1) as f:
  with open('hypp.txt', 'w') as f2:
      with open(f'reff.txt', 'w') as f3:
        for cols in csv.reader(f, delimiter='\t'):
            # print("hyp", cols[1]) #生成コード
            # print("ref", cols[2]) #正解コード
            f2.write(cols[1] + '\n')
            f3.write(cols[2] + '\n')
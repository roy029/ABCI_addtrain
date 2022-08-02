# ABCI_addtrain

GCN(RNN)のテストファイル
  てふさんの日本語-to-Python --> CoNaLa 英語-to-Python　に書き換え(できてるか微妙)

codebleu
  ABCI上で一応動くのを確認しました
  
  モジュールをインポート
  ```
  module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1
  ```
  CodeBLEU/parserの階層で、tree-sitter-pythonをgit cloneする
  
  [tree-sitterのGithub URL] https://github.com/tree-sitter/py-tree-sitter
  
  ★不安ポイント：tree-sitterは元のJavaからPythonに変えたけど、それ以外は特に変えていないこと...
  ```
  git clone https://github.com/tree-sitter/tree-sitter-python
  ```
  
  実行スクリプト(calc_code_bleu.pyがあるCodeBLEUの階層で実行する)
  ```
  python3 calc_code_bleu.py --refs Testfile/reff.txt --hyp Testfile/hypp.txt --lang python
  ```
  
  評価する正解・生成コードはCodeBLEU/Testfile/に、各々テキストファイル形式で与える
  ```
  hypp.txt(生成コード)  reff.txt(正解コード)
  ```


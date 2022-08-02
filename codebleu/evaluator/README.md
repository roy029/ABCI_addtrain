## codebleu
  ABCI上で一応それっぽい数字が出るのを確認した段階
  
  - モジュールをインポート
  ```
  module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1
  ```
  - CodeBLEU/parserの階層で、tree-sitter-pythonをgit cloneする
  
  [tree-sitterのGithub URL] https://github.com/tree-sitter/py-tree-sitter
  
  ★不安ポイント：syntax_match.pyにて、tree-sitterは元のJavaからPythonに変えたけど、それ以外は特に変えていないこと...
  ```
  git clone https://github.com/tree-sitter/tree-sitter-python
  ```
  
  - 実行スクリプト(calc_code_bleu.pyがあるCodeBLEUの階層で実行する)
  ```
  python3 calc_code_bleu.py --refs Testfile/reff.txt --hyp Testfile/hypp.txt --lang python
  ```
  
  - 評価する正解・生成コードはCodeBLEU/Testfile/に、各々テキストファイル形式で与える
  ```
  hypp.txt(生成コード)  reff.txt(正解コード)
  ```
  - CodeBLEU/keywords/python.txt
  
  ★不安ポイント：いったい何を与えるのか謎(現状空のファイルです)

### FIXME
  - 検証
  
    正しくBLEU値が出せているのか、CodeBLEUの正解・生成コードがあったら検証したい(keywordファイルがないから半ば諦め)
  - keyword/python.txtの作成
  
    論文を読んでみると、keyword(メソッドなど)が正しく出せていたら高いスコアを与えるっぽい。
    
    pythonコーパスから自動抽出できるようにしないとな...と考え中
  - 入力ファイルをCSV/TSVにしたい
    
    現状は、テキストファイルでそれぞれ与える必要がある --> とりあえずTestfile/tsv2txt.pyで2つのファイルを生成できるようにしました

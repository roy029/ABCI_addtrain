import csv
import unicodedata
# Unicode のテキスト処理を行うため
import string
# 一般的な文字列操作
import re
# 正規表現操作
import random
# 乱数を生成
import torch
# torch基本モジュール
import torch.nn as nn
# ネットワーク構築用
from torch import optim
# SGDを使うため
import torch.nn.functional as F
# ネットワーク用関数
import codecs
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from janome.tokenizer import Tokenizer
# Janomeのロード

#変更

# Python : 空白で単語分割
class Lang(object):
    def __init__(self, name):
        self.name = name
        # 登録する単語
        self.word2index = {}
        # 辞書を作成
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        # 0, 1番目はSOS, EOS
        # Start(End) Of Sequence
        self.n_words = 2
        # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            # sentenceを空白で区切り単語化した中にwordがあった時
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
        # もしwordが word2index辞書内にない場合
            self.word2index[word] = self.n_words
            # word を key としてその値に n_words をいれる
            self.word2count[word] = 1
            # word を key としてその値を1とする
            self.index2word[self.n_words] = word
            # n_words を key としてその値に word をいれる
            self.n_words += 1
        else:
        # wordが word2index辞書内にある場合
            self.word2count[word] += 1
            # word を key とした値に1を足す

# 日本語 : Janomeで単語分割
class JLang(Lang):
    def addJSentence(self, sentence):
        tokens = Tokenizer().tokenize(sentence, wakati = True)
        # 入力文を分かち書きし、文字列のリストとする
        for word in tokens :
            print("word:", word)
            self.addWord(word)

# #英語のトークナイザ導入：fixme
# class 

# 正規化
class Normalize(object):
    # def __init__(self, s):
    #     self.s = s

    def normalizeString(self, s):
        s = re.sub(r"\n", r" ", s)
        s = re.sub(r"\t+", r" ", s)
        # + 1回以上の繰り返し
        s = re.sub(r" {2,}", r" ", s)
        # {x, y} x回以上、y回以下の繰り返し
        s = re.sub(r"<SOS>", r"", s)
        return s

class Pair(object):
    def __init__(self):
        self.lines = []
        self.pairs = []

    def readLangs(self, lang1, lang2, reverse) :
        # ファイルを読み込んで行に分割する
        self.lines = codecs.open(txt, encoding='utf-8').read().strip().split('<EOS>')
        del self.lines[len(self.lines) - 1]
        # すべての行をペアに分割して正規化する
        self.pairs = [[Normalize().normalizeString(s) for s in l.split('<tab>')] for l in self.lines]
        # ペアを反転させ、Langインスタンスを作る
        if reverse :
        # もし reverse = False なら
            self.pairs = [list(reversed(p)) for p in self.pairs]
            # ペアを反転する
            input_lang = JLang(lang2)
            output_lang = Lang(lang1)
            # Jpn2Py
        else:
            input_lang = Lang(lang1)
            output_lang = JLang(lang2)
            # Py2Jpn
        return input_lang, output_lang, self.pairs

    def prepareData(self, lang1, lang2, reverse):
        input_lang, output_lang, self.pairs = self.readLangs(lang1, lang2, reverse)
        # print("Read {0} sentence pairs".format(len(pairs)))
        # print("Trimmed to {0} sentence pairs".format(len(pairs)))
        for pair in self.pairs:
            if len(pair) >= 2 :
                if reverse :
                # もしreverse = False なら
                    input_lang.addJSentence(pair[0])
                    output_lang.addSentence(pair[1])
                else :
                    input_lang.addSentence(pair[0])
                    output_lang.addJSentence(pair[1])
        # print("Counted words:")
        # print(input_lang.name, input_lang.n_words)
        # print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, self.pairs, reverse

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        # 行が各単語ベクトル、列が埋め込みの次元である行列を生成
        # Embedding(扱う単語の数, 隱れ層のサイズ(埋め込みの次元))
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        # 1 x 1 x n 型にベクトルサイズを変える
        # n の値は自動的に設定される
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
        # output が各系列のGRUの隱れ層ベクトル

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 線形結合を計算
        # hidden_size * 2
        # → 各系列のGRUの隠れ層とAttention層で計算したコンテキストベクトルをtorch.catでつなぎ合わせることで長さが２倍になる
        self.dropout = nn.Dropout(self.dropout_p)
        # 過学習の回避
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 列方向を確率変換したいから dim = 1
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # bmm でバッチも考慮してまとめて行列計算
        # ここでバッチが考慮されるから unsqueeze(0) が必要

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
    # コンテキストベクトルをまとめるための入れ物を用意する
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Time(object):
    def __init__(self):
        pass

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '{0}m {1}s'.format(m, int(s))

    def timeSince(self, since, percent):
        now = time.time()
        ns = now - since
        es = ns / (percent)
        rs = es - ns
        return '{0} (- {1})'.format(self.asMinutes(ns), self.asMinutes(rs))

class  Plot(object):
    def __init__(self):
        # plt.switch_backend('agg')
        # ↑あると表示されない
        pass

    def showPlot(self, points):
        # plt.figure()
        fig, ax = plt.subplots(1, 1)
        loc = ticker.MultipleLocator(base=0.2)
        # loc は定期的に ticker を配置する
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        # plt.savefig("plot.png")
        # content フォルダ下に保存される
        plt.show()

class Sentence(object):
    def __init__(self, lang, sentence):
        self.lang = lang
        self.sentence = sentence

    def indexesFromSentence(self):
        return [self.lang.word2index[word] for word in self.sentence.split(' ')]

    def indexesFromJSentence(self):
        tokens = Tokenizer().tokenize(self.sentence, wakati = True)
        # 入力文を分かち書きし、文字列のリストとする
        return [self.lang.word2index[word] for word in tokens]

    def tensorFromSentence(self):
        indexes = self.indexesFromSentence()
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorFromJSentence(self):
        indexes = self.indexesFromJSentence()
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class Tensor(object):
    def __init__(self, pair, reverse):
        self.pair = pair
        self.reverse = reverse

    def tensorsFromPair(self):
        if self.reverse :
            input_tensor = Sentence(input_lang, self.pair[0]).tensorFromJSentence()
            target_tensor = Sentence(output_lang, self.pair[1]).tensorFromSentence()
        else :
            input_tensor = Sentence(input_lang, self.pair[0]).tensorFromSentence()
            target_tensor = Sentence(output_lang, self.pair[1]).tensorFromJSentence()
        return (input_tensor, target_tensor)

class Train(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def train(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion, max_length=100, teacher_forcing_ratio=0.5):
        encoder_hidden = self.encoder.initHidden()
        # 勾配の初期化
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # データをテンソルに変換する
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
        # 教師強制を使用する場合
        # 教師強制 : 次の入力としてターゲットを送る
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
        else:
         # 教師強制の使用をしない : 次の入力として独自の予測値を使用する
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                # 入力として履歴から分ける
                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()     # 誤差逆伝播
        # パラメータの更新
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def trainIters(self, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0    # print_every ごとにリセットする
        plot_loss_total = 0     # plot_every ごとにリセットする

        # 最適化
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        training_pairs = [Tensor(random.choice(pairs), reverse).tensorsFromPair() for i in range(n_iters)]
        criterion = nn.NLLLoss()    # 損失関数

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = '{:.4f}'.format(print_loss_total / print_every)
                print_loss_total = 0
                print('{0} ({1} {2}%) {3}'.format(Time().timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        Plot().showPlot(plot_losses)

    def evaluate(self, reverse, sentence, max_length=100):
        with torch.no_grad():
            if reverse :
                input_tensor = Sentence(input_lang, sentence).tensorFromJSentence()
            else :
                input_tensor = Sentence(input_lang, sentence).tensorFromSentence()
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)
            # output_length =  encoder_outputs.size()[0]

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)      # SOS
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(reverse, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def evaluateOnce(self, s_list):
        out_list = []
        for s in s_list:
            norm_s = Normalize().normalizeString(s)
            s_list[s_list.index(s)] = norm_s

            print('{0}/{1}'.format(s_list.index(norm_s)+1, len(s_list)))
            print('>', norm_s)
            output_words, attentions = self.evaluate(reverse, norm_s)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            out_list.append(output_sentence)

def _main():
    s_list = []
    with open('rnn_test.txt', 'r') as f : #テストデータ
        for l in f :
            s_list.append(l[:-1])

    print('学習中……')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt = 'rnn_train.txt' #学習用データ
    SOS_token = 0
    EOS_token = 1

    # Jpn2Py
    input_lang, output_lang, pairs, reverse = Pair().prepareData('ja', 'py', True)
    # Py2Jpn
    # input_lang, output_lang, pairs, reverse = Pair().prepareData('py', 'ja', False)

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)


    Train(encoder1, attn_decoder1).trainIters(10000, print_every=500)

    print('翻訳中……')
    # Train(encoder1, attn_decoder1).evaluateRandomly()
    Train(encoder1, attn_decoder1).evaluateOnce(s_list)
    print("out_listの中身", out_list)

    # 梶浦メモ：出力形式を直す。

if __name__ == '__main__':
    _main()
"""
filename: lam.py : Lexicon, audio, and motor model for 2022jnps
date: 2022_0511
"""

import os
import sys
import typing
import numpy as np
import random
import pandas as pd
import gzip
from termcolor import colored
import jaconv
import re

import random
import torch

# from tqdm import tqdm         
#commandline で実行時
from tqdm.notebook import tqdm  #jupyter で実行時

import IPython
isColab = 'google.colab' in str(IPython.get_ipython())

# リソースの選択（CPU/GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MakeVocabDataset(_dict:dict, vocab=None)->dict:
    """上記 VOCAB を用いた下請け関数
    読み，音韻，モーラなどの情報を作成してデータセットといしての体裁を整える
    """
    _data = {}
    if vocab == None:
        vocab = VOCAB()
    x = [x[0] for x in _dict.values()]
    for _x in x:
        i = len(_data)  # 連番の番号を得る
        orth = vocab.ntt_orth2hira[_x] if _x in vocab.ntt_orth2hira else _x
        _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r = vocab.get7lists_from_orth(orth)
        phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r = vocab.get6ids(_phon, _orth, _yomi)
        _yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls = vocab.yomi2mora_transform(_yomi)    
        _data[i] = {'orig': orth, 'yomi': _yomi, 'ortho':_orth, 'ortho_ids': orth_ids, 
                    'ortho_r': _orth_r, 'ortho_ids_r': orth_ids_r, 
                    'phone':_phon, 'phone_ids': phon_ids, 'phone_r': _phon_r, 'phone_ids_r': phon_ids_r, 
                    'mora': _mora1, 'mora_r': _mora1_r, 'mora_ids': _mora_ids, 'mora_p': _mora_p, 
                    'mora_p_r': _mora_p_r, 'mora_p_ids': _mora_p_ids, 'mora_p_ids_r': _mora_p_ids_r,
                   }    

    return _data



def make_ntt_freq_data():
    """NTT日本語語彙特性の頻度情報を読み込んで，頻度辞書と表記から平仮名を返す辞書を返す"""
    print('# NTT日本語語彙特性 (天野，近藤; 1999, 三省堂)より頻度情報を取得')

    #データファイルの保存してあるディレクトリの指定
    ntt_dir = 'ccap'
    psy71_fname = 'psylex71utf8.txt'  # ファイル名
    #print('# 頻度情報の取得')
    #print('# psylex71.txt を utf8 に変換したファイルを用いる')
    with open(os.path.join(ntt_dir,psy71_fname), 'r') as f:
        ntt71raw = f.readlines()

    tmp = [line.split(' ')[:6] for line in ntt71raw]
    tmp2 = [[int(line[0]),line[2],line[4],int(line[5]), line[3]] for line in tmp]
    #単語ID(0), 単語，品詞，頻度 だけ取り出す
    ntt_freq = {x[0]-1:{'単語':jaconv.normalize(x[1]),
                        '品詞':x[2],
                        '頻度':x[3], 
                        'よみ':jaconv.kata2hira(jaconv.normalize(x[4]))
                        } for x in tmp2}
    #ntt_freq = {x[0]-1:{'単語':x[1],'品詞':x[2],'頻度':x[3], 'よみ':x[4]} for x in tmp2}
    ntt_orth2hira = {ntt_freq[x]['単語']:ntt_freq[x]['よみ'] for x in ntt_freq}
    #print(f'#登録総単語数: {len(ntt_freq)}')

    Freq = np.zeros((len(ntt_freq)), dtype=np.uint)  #ソートに使用する numpy 配列
    for i, x in enumerate(ntt_freq):
        Freq[i] = ntt_freq[i]['頻度']

    Freq_sorted = np.argsort(Freq)[::-1]  #頻度降順に並べ替え

    # self.ntt_freq には頻度順に単語が並んでいる
    return [ntt_freq[x]['単語']for x in Freq_sorted], ntt_orth2hira


class VOCAB():
    '''
    訓練データとしては，NTT 日本語語彙特性 (天野，近藤, 1999, 三省堂) の頻度データ，実際のファイル名としては `pslex71.txt` から頻度データを読み込んで，高頻度語を訓練データとする。
    ただし，検証データに含まれる単語は訓練データとして用いない。

    検証データとして，以下のいずれかを考える
    1. TLPA (藤田 他, 2000, 「失語症語彙検査」の開発，音声言語医学 42, 179-202) 
    2. SALA 上智大学失語症語彙検査 

    このオブジェクトクラスでは，
    `phone_vocab`, `ortho_vocab`, `ntt_freq`, に加えて，単語の読みについて ntt_orth2hira によって読みを得ることにした。

    * `train_data`, `test_data` という辞書が本体である。
    各辞書の項目には，さらに
    `Vocab_ja.test_data[0].keys() = dict_keys(['orig', 'ortho', 'phone', 'ortho_ids', 'phone_ids', 'semem'])`

    各モダリティ共通トークンとして以下を設定した
    * <PAD>: 埋め草トークン
    * <EQW>: 単語終端トークン
    * <SOW>: 単語始端トークン
    * <UNK>: 未定義トークン

    このクラスで定義されるデータは 2 つの辞書である。すなわち 1. train_data, 2. tlpa_data である。
    各辞書は，次のような辞書項目を持つ。
    ```
    {0: {'orig': 'バス',
    'yomi': 'ばす',
    'ortho': ['バ', 'ス'],
    'ortho_ids': [695, 514],
    'ortho_r': ['ス', 'バ'],
    'ortho_ids_r': ['ス', 'バ'],
    'phone': ['b', 'a', 's', 'u'],
    'phone_ids': [23, 7, 19, 12],
    'phone_r': ['u', 's', 'a', 'b'],
    'phone_ids_r': [12, 19, 7, 23],
    'mora': ['ば', 'す'],
    'mora_r': ['す', 'ば'],
    'mora_ids': [87, 47],
    'mora_p': ['b', 'a', 's', 'u'],
    'mora_p_r': ['s', 'u', 'b', 'a'],
    'mora_p_ids': [6, 5, 31, 35],
    'mora_p_ids_r': [31, 35, 6, 5]},
    ```
    '''
    
    def __init__(self, 
                 traindata_size = 10000,
                 w2v=None,
                 yomi=None,
                 #test_name='TLPA',  # or 'SALA'
                ):

        #isColab = 'google.colab' in str(IPython.get_ipython())

        if w2v != None:
            self.w2v = w2v
        else:
            from ccap import ccap_w2v
            #from ccap_w2v import ccap_w2v
            self.w2v = ccap_w2v(is2017=False).w2v

        if yomi != None:
            self.yomi = yomi
        else:
            #from mecab_settings import yomi
            from ccap.mecab_settings import yomi
            self.yomi = yomi

        # if wakati != None:
        #     self.wakati = wakati
        # else:
        #     from ccap.mecab_settings import wakati
        #     self.wakati = wakati

        # 訓練語彙数の上限 `training_size` を設定 
        self.traindata_size = traindata_size

        # `self.moraWakachi()` で用いる正規表現のあつまり 各条件を正規表現で表す
        self.c1 = '[うくすつぬふむゆるぐずづぶぷゔ][ぁぃぇぉ]' #ウ段＋「ァ/ィ/ェ/ォ」
        self.c2 = '[いきしちにひみりぎじぢびぴ][ゃゅぇょ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        self.c3 = '[てで][ぃゅ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        self.c4 = '[ぁ-ゔー]' #カタカナ１文字（長音含む）
        self.c5 = '[ふ][ゅ]'
        # self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        # self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        # self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        # self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
        #cond = '('+c1+'|'+c2+'|'+c3+'|'+c4+')'
        self.cond = '('+self.c5+'|'+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+')'
        self.re_mora = re.compile(self.cond)
        # 以上 `self.moraWakachi()` で用いる正規表現の定義


        self.ortho_vocab, self.ortho_freq = ['<PAD>', '<EOW>','<SOW>','<UNK>'], {}
        self.phone_vocab, self.phone_freq = ['<PAD>', '<EOW>','<SOW>','<UNK>'], {}
        self.phone_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>', \
        'n', 'o', 'h', 'a', 'i', 't', 'g', 'r', 'u', 'd', 'e', 'sh', 'q', 'm', 'k', 's', 'y', 'p', 'N', 'b', 'ts', 'o:',\
        'ky', 'f', 'w', 'ch', 'ry', 'gy', 'u:', 'z', 'j', 'py', 'hy', 'i:', 'e:', 'a:', 'by', 'ny', 'my', 'dy', \
        'a::', 'u::', 'o::']

        # 全モーラリストを `mora_vocab` として登録
        self.mora_vocab=['<PAD>', '<EOW>', '<SOW>', '<UNK>', \
        'ぁ', 'あ', 'ぃ', 'い', 'ぅ', 'う', 'うぃ', 'うぇ', 'うぉ', 'ぇ', 'え', 'お', \
        'か', 'が', 'き', 'きゃ', 'きゅ', 'きょ', 'ぎ', 'ぎゃ', 'ぎゅ', 'ぎょ', 'く', 'くぁ', 'くぉ', 'ぐ', 'ぐぁ', 'け', 'げ', 'こ', 'ご', \
        'さ', 'ざ', 'し', 'しぇ', 'しゃ', 'しゅ', 'しょ', 'じ', 'じぇ', 'じゃ', 'じゅ', 'じょ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ', \
        'た', 'だ', 'ち', 'ちぇ', 'ちゃ', 'ちゅ', 'ちょ', 'ぢ', 'ぢゃ', 'ぢょ', 'っ', 'つ', 'つぁ', 'つぃ', 'つぇ', 'つぉ', 'づ', 'て', 'てぃ', 'で', 'でぃ', 'でゅ', 'と', 'ど', \
        'な', 'に', 'にぇ', 'にゃ', 'にゅ', 'にょ', 'ぬ', 'ね', 'の', \
        'は', 'ば', 'ぱ', 'ひ', 'ひゃ', 'ひゅ', 'ひょ', 'び', 'びゃ', 'びゅ', 'びょ', 'ぴ', 'ぴゃ', 'ぴゅ', 'ぴょ', 'ふ', 'ふぁ', 'ふぃ', 'ふぇ', 'ふぉ', 'ふゅ', 'ぶ', 'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ', \
        'ま', 'み', 'みゃ', 'みゅ', 'みょ', 'む', 'め', 'も', \
        'や', 'ゆ', 'よ', 'ら', 'り', 'りゃ', 'りゅ', 'りょ', 'る', 'れ', 'ろ', 'ゎ', 'わ', 'ゐ', 'ゑ', 'を', 'ん', 'ー']

        # モーラから音への変換表を表す辞書を `mora2jul` として登録
        self.mora2jul={'ぁ': ['a'], 'あ': ['a'], 'ぃ': ['i'], 'い': ['i'], 'ぅ': ['u'], 'う': ['u'], 'うぃ': ['w', 'i'], 'うぇ': ['w', 'e'], 'うぉ': ['w', 'o'], \
        'ぇ': ['e'], 'え': ['e'], 'お': ['o'], 'か': ['k', 'a'], 'が': ['g', 'a'], 'き': ['k', 'i'], 'きゃ': ['ky', 'a'], 'きゅ': ['ky', 'u'], 'きょ': ['ky', 'o'], \
        'ぎ': ['g', 'i'], 'ぎゃ': ['gy', 'a'], 'ぎゅ': ['gy', 'u'], 'ぎょ': ['gy', 'o'], 'く': ['k', 'u'], 'くぁ': ['k', 'u', 'a'], 'くぉ': ['k', 'u', 'o'], 'ぐ': ['g', 'u'], \
        'ぐぁ': ['g', 'u', 'a'], 'け': ['k', 'e'], 'げ': ['g', 'e'], 'こ': ['k', 'o'], 'ご': ['g', 'o'], 'さ': ['s', 'a'], 'ざ': ['z', 'a'], 'し': ['sh', 'i'], \
        'しぇ': ['sh', 'e'], 'しゃ': ['sh', 'a'], 'しゅ': ['sh', 'u'], 'しょ': ['sh', 'o'], 'じ': ['j', 'i'], 'じぇ': ['j', 'e'], 'じゃ': ['j', 'a'], 'じゅ': ['j', 'u'], \
        'じょ': ['j', 'o'], 'す': ['s', 'u'], 'ず': ['z', 'u'], 'せ': ['s', 'e'], 'ぜ': ['z', 'e'], 'そ': ['s', 'o'], 'ぞ': ['z', 'o'], 'た': ['t', 'a'], 'だ': ['d', 'a'], \
        'ち': ['ch', 'i'], 'ちぇ': ['ch', 'e'], 'ちゃ': ['ch', 'a'], 'ちゅ': ['ch', 'u'], 'ちょ': ['ch', 'o'], 'ぢ': ['j', 'i'], 'ぢゃ': ['j', 'a'], 'ぢょ': ['j', 'o'], \
        'っ': ['q'], 'つ': ['ts', 'u'], 'つぁ': ['ts', 'a'], 'つぃ': ['ts', 'i'], 'つぇ': ['ts', 'e'], 'つぉ': ['ts', 'o'], 'づ': ['z', 'u'], 'て': ['t', 'e'], 'てぃ': ['t', 'i'], \
        'で': ['d', 'e'], 'でぃ': ['d', 'i'], 'でゅ': ['dy', 'u'], 'と': ['t', 'o'], 'ど': ['d', 'o'], 'な': ['n', 'a'], 'に': ['n', 'i'], 'にぇ': ['n', 'i', 'e'], \
        'にゃ': ['ny', 'a'], 'にゅ': ['ny', 'u'], 'にょ': ['ny', 'o'], 'ぬ': ['n', 'u'], 'ね': ['n', 'e'], 'の': ['n', 'o'], 'は': ['h', 'a'], 'ば': ['b', 'a'], \
        'ぱ': ['p', 'a'], 'ひ': ['h', 'i'], 'ひゃ': ['hy', 'a'], 'ひゅ': ['hy', 'u'], 'ひょ': ['hy', 'o'], 'び': ['b', 'i'], 'びゃ': ['by', 'a'], 'びゅ': ['by', 'u'], \
        'びょ': ['by', 'o'], 'ぴ': ['p', 'i'], 'ぴゃ': ['py', 'a'], 'ぴゅ': ['py', 'u'], 'ぴょ': ['py', 'o'], 'ふ': ['f', 'u'], 'ふぁ': ['f', 'a'], 'ふぃ': ['f', 'i'], \
        'ふぇ': ['f', 'e'], 'ふぉ': ['f', 'o'], 'ふゅ': ['hy', 'u'], 'ぶ': ['b', 'u'], 'ぷ': ['p', 'u'], 'へ': ['h', 'e'], 'べ': ['b', 'e'], 'ぺ': ['p', 'e'], 'ほ': ['h', 'o'], \
        'ぼ': ['b', 'o'], 'ぽ': ['p', 'o'], 'ま': ['m', 'a'], 'み': ['m', 'i'], 'みゃ': ['my', 'a'], 'みゅ': ['my', 'u'], 'みょ': ['my', 'o'], 'む': ['m', 'u'], 'め': ['m', 'e'], \
        'も': ['m', 'o'], 'や': ['y', 'a'], 'ゆ': ['y', 'u'], 'よ': ['y', 'o'], 'ら': ['r', 'a'], 'り': ['r', 'i'], 'りゃ': ['ry', 'a'], 'りゅ': ['ry', 'u'], 'りょ': ['ry', 'o'], \
        'る': ['r', 'u'], 'れ': ['r', 'e'], 'ろ': ['r', 'o'], 'ゎ': ['w', 'a'], 'わ': ['w', 'a'], 'ゐ': ['i'], 'ゑ': ['e'], 'を': ['o'], 'ん': ['N'], 'ー':[':']}

        # モーラに用いる音を表すリストを `mora_p_vocab` として登録
        self.mora_p_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',  \
        'N', 'a', 'b', 'by', 'ch', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', \
        'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'u', 'w', 'y', 'z']

        # 母音を表す音から ひらがな への変換表を表す辞書を `vow2hira` として登録
        self.vow2hira = {'a':'あ', 'i':'い', 'u':'う', 'e':'え', 'o':'お', 'N':'ん'}

        self.mora_freq = {'<PAD>':0, '<EOW>':0, '<SOW>':0, '<UNK>':0}
        self.mora_p = {}

        # NTT 日本語語彙特性データから，`self.train_data` を作成
        self.ntt_freq, self.ntt_orth2hira = self.make_ntt_freq_data()

        # if test_name == 'TLPA':                      # TLPA データを読み込み
        #     self.test_name = test_name
        #     self.test_vocab = self.get_tlpa_vocab()  # tlpa データを読み込み
        # elif test_name == 'SALA':                    # sala データの読み込み
        #     self.test_name = test_name
        #     self.test_vocab = self.get_sala_vocab() # sala データを読み込む
        # else:
        #     self.test_name = None
        #     print(f'Invalid test_name:{test_name}')
        #     sys.exit()

        self.ntt_freq_vocab = self.set_train_vocab()
        self.train_data, self.excluded_data = {}, []
        max_ortho_length, max_phone_length, max_mora_length, max_mora_p_length = 0, 0, 0, 0

        for orth in tqdm(self.ntt_freq_vocab):
            i = len(self.train_data)

            # 書記素 `orth` から 読みリスト，音韻表現リスト，音韻表現反転リスト，
            # 書記表現リスト，書記表現反転リスト，モーラ表現リスト，モーラ表現反転リスト の 7 つのリストを得る
            _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r = self.get7lists_from_orth(orth)

            # 音韻語彙リスト `self.phone_vocab` に音韻が存在していれば True そうでなければ False というリストを作成し，
            # そのリスト無いに False があれば，排除リスト `self.excluded_data` に登録する
            if False in [True if p in self.phone_vocab else False for p in _phon]:
                self.excluded_data.append(orth)
                continue

            phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r = self.get6ids(_phon, _orth, _yomi)
            _yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls = self.yomi2mora_transform(_yomi)
            self.train_data[i] = {'orig': orth, 'yomi': _yomi,
                                 'ortho':_orth, 'ortho_ids': orth_ids, 'ortho_r': _orth_r, 'ortho_ids_r': orth_ids_r,
                                 'phone':_phon, 'phone_ids': phon_ids, 'phone_r': _phon_r, 'phone_ids_r': phon_ids_r,
                                 'mora': _mora1, 'mora_r': _mora1_r, 'mora_ids': _mora_ids, 'mora_p': _mora_p,
                                 'mora_p_r': _mora_p_r, 'mora_p_ids': _mora_p_ids, 'mora_p_ids_r': _mora_p_ids_r,
                                 #'semem':self.w2v[orth],
                               }
            len_orth, len_phon, len_mora, len_mora_p = len(_orth), len(_phon), len(_mora), len(_mora_p)
            max_ortho_length = len_orth if len_orth > max_ortho_length else max_ortho_length
            max_phone_length = len_phon if len_phon > max_phone_length else max_phone_length
            max_mora_length = len_mora if len_mora > max_mora_length else max_mora_length
            max_mora_p_length = len_mora_p if len_mora_p > max_mora_p_length else max_mora_p_length
            if len(self.train_data) >= self.traindata_size: # 上限値に達したら終了する
                self.train_vocab = [self.train_data[x]['orig'] for x in self.train_data]
                break 

        self.max_ortho_length = max_ortho_length
        self.max_phone_length = max_phone_length
        self.max_mora_length = max_mora_length
        self.max_mora_p_length = max_mora_p_length

    def yomi2mora_transform(self, yomi):
        """ひらがな表記された引数 `yomi` から，日本語の 拍(モーラ)  関係のデータを作成する
        引数:
        yomi:str ひらがな表記された単語 UTF-8 で符号化されていることを仮定している

        戻り値:
        yomi:str 入力された引数
        _mora1:list[str] `_mora` に含まれる長音 `ー` を直前の母音で置き換えた，モーラ単位の分かち書きされた文字列のリスト
        _mora1_r:list[str] `_mora1` を反転させた文字列リスト
        _mora:list[str] `self.moraWakatchi()` によってモーラ単位で分かち書きされた文字列のリスト
        _mora_ids:list[int] `_mora` を対応するモーラ ID で置き換えた整数値からなるリスト
        _mora_p:list[str] `_mora` を silius によって音に変換した文字列リスト
        _mora_p_r:list[str] `_mora_p` の反転リスト
        _mora_p_ids:list[int] `mora_p` の各要素を対応する 音 ID に変換した数値からなるリスト
        _mora_p_ids_r:list[int] `mora_p_ids` の各音を反転させた数値からなるリスト
        _juls:list[str]: `yomi` を julius 変換した音素からなるリスト
        """
        _mora = self.moraWakachi(yomi) # 一旦モーラ単位の分かち書きを実行して `_mora` に格納
    
        # 単語をモーラ反転した場合に長音「ー」の音が問題となるので，長音「ー」を母音で置き換えるためのプレースホルダとして. `_mora` を用いる
        _mora1 = _mora.copy()     

        # その他のプレースホルダの初期化，モーラ，モーラ毎 ID, モーラ音素，モーラの音素の ID， モーラ音素の反転，モーラ音素の反転 ID リスト
        _mora_ids, _mora_p, _mora_p_ids, _mora_p_r, _mora_p_ids_r = [], [], [], [], []
        _m0 = 'ー' # 長音記号
    
        for i, _m in enumerate(_mora): # 各モーラ単位の処理と登録
        
            __m = _m0 if _m == 'ー' else _m               # 長音だったら，前音の母音を __m とし，それ以外は自分自身を __m に代入
            _mora1[i] = __m                               # 長音を変換した結果を格納
            _mora_ids.append(self.mora_vocab.index(__m))  # モーラを ID 番号に変換
            _mora_p += self.mora2jul[__m]                 # モーラを音素に変換して `_mora_p` に格納
        
            # 変換した音素を音素 ID に変換して，`_mora_p_ids` に格納
            _mora_p_ids += [self.mora_p_vocab.index(_p) for _p in self.mora2jul[__m]] 
        
            if not _m in self.mora_freq: # モーラの頻度表を集計
                self.mora_freq[__m] = 1
            else:
                self.mora_freq[__m] +=1
            
            if self.hira2julius(__m)[-1] in self.vow2hira:           # 直前のモーラの最終音素が母音であれば
                _m0 = self.vow2hira[self.hira2julius(__m)[-1]]  # 直前の母音を代入しておく。この処理が 2022_0311 でのポイントであった

        # モーラ分かち書きした単語 _mora1 の反転を作成し `_mora1_r` に格納
        _mora1_r = [m for m in _mora1[::-1]]
    
        for _m in _mora1_r:                   # 反転した各モーラについて
            # モーラ単位で julius 変換して音素とし `_mora_p_r` に格納
            _mora_p_r += self.mora2jul[_m]

            # `_mora_p_r` に格納した音素を音素 ID に変換し `_mora_p_ids` に格納
            _mora_p_ids_r += [self.mora_p_vocab.index(_p) for _p in self.mora2jul[_m]]

        _juls = self.hira2julius(yomi)
        
        return yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls


    def get6ids(self, _phon, _orth, yomi):

        # 音韻 ID リスト `phone_ids` に音素を登録する
        phone_ids = [self.phone_vocab.index(p) for p in _phon]

        # 直上の音韻 ID リストの逆転を作成
        phone_ids_r = [p_id for p_id in phone_ids[::-1]]

        # 書記素 ID リスト `ortho_ids` に書記素を登録
        for o in _orth:
            if not o in self.ortho_vocab:
                self.ortho_vocab.append(o)
        orth_ids = [self.ortho_vocab.index(o) for o in _orth]

        # 直上の書記素 ID リストの逆転を作成
        orth_ids_r = [o_id for o_id in orth_ids[::-1]]
        #orth_ids_r = [o_id for o_id in _orth[::-1]]

        mora_ids = []
        for _p in self.hira2julius(yomi):
            mora_ids.append(self.phone_vocab.index(_p))

        mora_ids_r = []
        #print(f'yomi:{yomi}, self.moraWakati(yomi):{self.moraWakachi(jaconv.hira2kata(yomi))}')
        # for _p in self.moraWakachi(jaconv.hira2kata(yomi))[::-1]:
        #     for __p in  self.phone_vocab.index(self.hira2julius(jaconv.kata2hira(_p))):
        #         mora_ids_r.append(__p)

        return phone_ids, phone_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r


    def moraWakachi(self, hira_text):
        """ ひらがなをモーラ単位で分かち書きする
        https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12"""

        return self.re_mora.findall(hira_text)


    def _kana_moraWakachi(kan_text):
        # self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        # self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        # self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        # self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）

        self.cond = '('+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+')'
        self.re_mora = re.compile(self.cond)
        return re_mora.findall(kana_text)



    def get7lists_from_orth(self, orth):
        """書記素 `orth` から 読みリスト，音韻表現リスト，音韻表現反転リスト，
        書記表現リスト，書記表現反転リスト，モーラ表現リスト，モーラ表現反転リスト の 7 つのリストを得る"""

        # 単語の表層形を，読みに変換して `_yomi` に格納
        # ntt_orth2hira という命名はおかしかったから修正 2022_0309
        if orth in self.ntt_orth2hira:
            _yomi = self.ntt_orth2hira[orth]
        else:
            _yomi = jaconv.kata2hira(self.yomi(orth).strip())

        # `_yomi` を julius 表記に変換して `_phon` に代入
        _phon = self.hira2julius(_yomi)# .split(' ')

        # 直上の `_phon` の逆転を作成して `_phone_r` に代入
        _phon_r = [_p_id for _p_id in _phon[::-1]]

        # 書記素をリストに変換
        _orth = [c for c in orth]

        # 直上の `_orth` の逆転を作成して `_orth_r` に代入
        _orth_r = [c for c in _orth[::-1]]

        _mora = self.moraWakachi(jaconv.hira2kata(_yomi))
        _mora_r = [_m for _m in _mora[::-1]]
        _mora = self.moraWakachi(_yomi)
        for _m in _mora:
            if not _m in self.mora_vocab:
                self.mora_vocab.append(_m)
            for _j in self.hira2julius(_m):
                if not _j in self.mora_p:
                    self.mora_p[_j] = 1
                else:
                    self.mora_p[_j] += 1

        return _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r



    def hira2julius(self, text:str)->str:
        """`jaconv.hiragana2julius()` では未対応の表記を扱う"""
        text = text.replace('ゔぁ', ' b a')
        text = text.replace('ゔぃ', ' b i')
        text = text.replace('ゔぇ', ' b e')
        text = text.replace('ゔぉ', ' b o')
        text = text.replace('ゔゅ', ' by u')

        #text = text.replace('ぅ゛', ' b u')
        text = jaconv.hiragana2julius(text).split()
        return text

    def __len__(self)->int:
        return len(self.data)

    def __call__(self, x:int)->dict:
        return self.data[x]

    def __getitem__(self, x:int)->dict:
        return self.data[x]

    def set_train_vocab(self):
    #def set_train_vocab_minus_test_vocab(self):
        """JISX2008-1990 コードから記号とみなしうるコードを集めて ja_symbols とする
        記号だけから構成されている word2vec の項目は排除するため
        """
        self.ja_symbols = '、。，．・：；？！゛゜´\' #+ \'｀¨＾‾＿ヽヾゝゞ〃仝々〆〇ー—‐／＼〜‖｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋−±×÷＝≠＜＞≦≧∞∴♂♀°′″℃¥＄¢£％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨¬⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪†‡¶◯#ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        self.ja_symbols_normalized = jaconv.normalize(self.ja_symbols)

        print(f'# 訓練に用いる単語の選定 {self.traindata_size} 語')
        vocab = []; i=0
        #while (len(vocab) < self.traindata_size) and (i<len(self.ntt_freq)):
        while i<len(self.ntt_freq):
            word = self.ntt_freq[i]
            if word == '\u3000': # NTT 日本語の語彙特性で，これだけ変なので特別扱い
                i += 1
                continue

            # 良い回避策が見つからないので，以下の行の変換だけ特別扱いしている
            word = jaconv.normalize(word).replace('・','').replace('ヴ','ブ')

            if (not word in self.ja_symbols) and (not word.isascii()) and (word in self.w2v):
                #if not word in self.test_vocab:
                #if not word in self.tlpa_vocab:
                vocab.append(word)
                if len(vocab) >= self.traindata_size:
                    return vocab
            i += 1
        return vocab


    def make_ntt_freq_data(self):
        print('# NTT日本語語彙特性 (天野，近藤; 1999, 三省堂)より頻度情報を取得')

        #データファイルの保存してあるディレクトリの指定
        ntt_dir = 'ccap'
        psy71_fname = 'psylex71utf8.txt'  # ファイル名
        #print('# 頻度情報の取得')
        #print('# psylex71.txt を utf8 に変換したファイルを用いる')
        with open(os.path.join(ntt_dir,psy71_fname), 'r') as f:
            ntt71raw = f.readlines()

        tmp = [line.split(' ')[:6] for line in ntt71raw]
        tmp2 = [[int(line[0]),line[2],line[4],int(line[5]), line[3]] for line in tmp]
        #単語ID(0), 単語，品詞，頻度 だけ取り出す

        ntt_freq = {x[0]-1:{'単語':jaconv.normalize(x[1]),
                            '品詞':x[2],
                            '頻度':x[3], 
                            'よみ':jaconv.kata2hira(jaconv.normalize(x[4]))
                            } for x in tmp2}
        #ntt_freq = {x[0]-1:{'単語':x[1],'品詞':x[2],'頻度':x[3], 'よみ':x[4]} for x in tmp2}
        ntt_orth2hira = {ntt_freq[x]['単語']:ntt_freq[x]['よみ'] for x in ntt_freq}
        #print(f'#登録総単語数: {len(ntt_freq)}')

        Freq = np.zeros((len(ntt_freq)), dtype=np.uint)  #ソートに使用する numpy 配列
        for i, x in enumerate(ntt_freq):
            Freq[i] = ntt_freq[i]['頻度']

        Freq_sorted = np.argsort(Freq)[::-1]  #頻度降順に並べ替え

        # self.ntt_freq には頻度順に単語が並んでいる
        return [ntt_freq[x]['単語']for x in Freq_sorted], ntt_orth2hira


# # 乱数シード固定（再現性の担保）
# seed = 42 # 参照 ダグラス・アダムス「銀河ヒッチハイカーズガイド」 The Hitchhiker's Guide to the Galaxy
# def fix_seed(seed):
#     random.seed(seed)  # for random
#     np.random.seed(seed) # for numpy
#     # for pytorch
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# fix_seed(seed)
# # データローダーのサブプロセスの乱数の seed が固定
# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)
# print(worker_init_fn(1))

import platform

def get_tlpa_naming_vocab(ntt_freq=None, ntt_orth2hira=None, w2v=None):
    """TLPA データの読み込み"""

    if ntt_freq == None or ntt_orth2hira == None:
        print('# 上間先生からいただいた TLPA 文字データ `呼称(TLPA)語彙属性.xlsx` を読み込む')
        ntt_freq, ntt_orth2hira = make_ntt_freq_data()

    if w2v == None:
        from ccap import ccap_w2v
        #from ccap_w2v import ccap_w2v
        w2v = ccap_w2v(is2017=False).w2v

    #ccap_base = 'ccap'

    HOSTNAME = platform.node().split('.')[0]
    HOME = '/Users/_asakawa' if HOSTNAME == 'Sinope' else '/Users/asakawa'
    ccap_base = os.path.join(HOME, 'study/2020ccap')
    uema_excel = '呼称(TLPA)語彙属性.xlsx'
    k = pd.read_excel(os.path.join(ccap_base, uema_excel), header=[1], sheet_name='刺激属性')
    tlpa_words = list(k['標的語'])

    # 標的語には文字 '／' が混じっているので，切り分けて別の単語と考える
    _tlpa_words = []
    for word in tlpa_words:
        for x in word.split('／'):
            _tlpa_words.append(x)

    # word2vec に存在するか否かのチェックと，存在するのであれば word2vec におけるその単語のインデックスを得る
    # word2vec では，単語は頻度順に並んでいるので最大値インデックスを取る単語が，最低頻度語ということになる
    # そのため，TLPA の刺激語のうち，最低頻度以上の全単語を使って学習を行うことを考えるためである。
    max_idx, __tlpa_words = 0, []
    for w in _tlpa_words:
        if w in w2v:
            idx = w2v.key_to_index[w]
            max_idx = idx if idx > max_idx else max_idx
            __tlpa_words.append(w)

    return __tlpa_words

def get_sala_naming_vocab(ntt_freq=None, ntt_orth2hira=None):
    """SALA データの読み込み"""

    if ntt_freq == None or ntt_orth2hira == None:
        ntt_freq, ntt_orth2hira = make_ntt_freq_data()

    HOSTNAME = platform.node().split('.')[0]
    HOME = '/Users/_asakawa' if HOSTNAME == 'Sinope' else '/Users/asakawa'
    ccap_base = os.path.join(HOME, 'study/2020ccap')
    #ccap_base = '/Users/_asakawa/study/2020ccap/notebooks/'
    with open(os.path.join(ccap_base,'notebooks/ccap/data/sala_data.txt'), 'r', encoding='utf-8') as f:
        p = f.readlines()

    sala, sala_vocab = {}, []
    for l in p:
        _l = l.strip().split(' ')
        tag1 = jaconv.normalize(_l[-1].split('（')[0])
        tag2 = _l[-1].split('（')[-1].replace('）','')
        yomi = ntt_orth2hira[tag1] if tag1 in ntt_orth2hira else None
        if yomi == None:
            if tag1 == '轆轤首':
                yomi = 'ろくろくび'
            elif tag1 == '擂粉木':
                yomi = 'すりこぎ'
            elif tag1 == '擂鉢':
                yomi = 'すりばち'
            elif tag1 == 'こま':
                yomi = 'こま'
            elif tag1 == 'おにぎり':
                yomi = 'おにぎり'
            sala_vocab.append(yomi)
        else:
            sala_vocab.append(tag1)
        
        sala[len(sala)] = {'tag': tag1, 
                        'code': _l[0], 
                        'feat': _l[1],
                        'yomi': yomi }
    return sala_vocab


def read_json_tlpa1234_sala_r29_30_31(json_fname='2022_0508SALA_TLPA.json'):
    import json

    # 作成済の json ファイルを読み込む
    csv_fnames = {'TLPA':'2022_0502TLPA.csv',
                  'SALA': '2022_0502sala.csv'}
    if os.path.exists(json_fname):
        with open(json_fname, 'r', encoding='utf-8') as f:
            TLPA1, TLPA2, TLPA3, TLPA4, SALA_R29, SALA_R30, SALA_R31 = json.load(f)
    else:
        print(f'The json file:{json_fname} does not exist')
        print(f'trying make a json file from csv files')
        data_dir = '..'                   # データの存在するディレクトリの指定
        TLPA_fname = csv_fnames['TLPA']   # `失語症語彙検査 TLPA II III IV.pdf` をテキスト化した csv ファイル名
        SALA_fname = csv_fnames['SALA']   # `Rall.pdf` SALA の 29-32 までの csv ファイル名。ただし R31, R32 は未入力

        # 語彙判断検査データの読み込み
        with open(os.path.join(data_dir, TLPA_fname), 'r', encoding='utf-8') as f:
            L = f.readlines()

        # csv ファイルから `#` で始まる行を読み込まない
        L = [_l.strip().split(',') for _l in L if _l[0] != '#']

        # 先頭要素をキーとして `TLPA` 辞書を作成
        TLPA = {l[0]:l[1:] for l in L}

        # TLPA 辞書を 4 つに分割
        TLPA1, TLPA2, TLPA3, TLPA4 = {}, {}, {}, {}
        for k in TLPA.keys():
            _k = k.split('_')[-1]
            if 'TLPA1_' in k:
                TLPA1[_k] = TLPA[k].copy()
            if 'TLPA2_' in k:
                TLPA2[_k] = TLPA[k].copy()
            if 'TLPA3_' in k:
                TLPA3[_k] = TLPA[k].copy()
            if 'TLPA4_' in k:
                TLPA4[_k] = TLPA[k].copy()


        # SALA R29, R30, R31 データの読み込み
        with open(os.path.join(data_dir, SALA_fname), 'r', encoding='utf-8') as f:
            R = f.readlines()

        # csv ファイルから `#` で始まる行を読み込まない
        R = [_R.strip().split(',') for _R in R if _R[0] != '#']

        # 先頭要素をキーとして `SALA` 辞書を作成
        SALA_ = {r[0]:r[1:] for r in R}

        for k in SALA_.keys():
            if 'R29_' in k: # SALA R29 は復唱なので，ひらがな表記していある カッコ内の単語を用いる
                # 余分なカッコを取り除く
                SALA_[k][0] = SALA_[k][0].split('（')[-1].replace('）','')

        # SALA_ 辞書を 3 つに分割
        SALA_R29, SALA_R30, SALA_R31 = {}, {}, {}
        for k in SALA_.keys():
            _k = k.split('_')[-1]
            if 'R29_' in k:
                SALA_R29[_k] = SALA_[k].copy()
            if 'R30_' in k:
                SALA_R30[_k] = SALA_[k].copy()
            if 'R31_' in k:
                SALA_R31[_k] = SALA_[k].copy()

        # 以上で，TLPA1, TLPA2, TLPA3, TLPA4, SALA_R29, SALA_R30, SALA_R31 データ辞書を作成。
        # いずれの辞書もキーは int である。values は list で，第一要素が刺激語である。
        # 第 2 要素以下は，刺激の属性となっている。

        # json ファイルとして保存
        with open(json_fname, 'w', encoding='utf-8') as f:
            json.dump((TLPA1, TLPA2, TLPA3, TLPA4, SALA_R29, SALA_R30, SALA_R31), f, ensure_ascii=False, indent=2)

        # 読み込みテスト
        with open(json_fname, 'r', encoding='utf-8') as f:
            _TLPA1, _TLPA2, _TLPA3, _TLPA4, _SALA_R29, _SALA_R30, _SALA_R31 = json.load(f)
        print(f'TLPA1 == _TLPA1: {TLPA1==_TLPA1}')
        print(f'TLPA2 == _TLPA2: {TLPA2==_TLPA2}')
        print(f'TLPA3 == _TLPA3: {TLPA3==_TLPA3}')
        print(f'TLPA4 == _TLPA4: {TLPA4==_TLPA4}')
        print(f'SALA_R29 == _SALA_R29: {SALA_R29 == _SALA_R29}')
        print(f'SALA_R30 == _SALA_R30: {SALA_R30 == _SALA_R30}')
        print(f'SALA_R31 == _SALA_R31: {SALA_R31 == _SALA_R31}')

    return TLPA1, TLPA2, TLPA3, TLPA4, SALA_R29, SALA_R30, SALA_R31


#def _make_vocab_dataset(_dict:dict, 
def make_vocab_dataset(_dict:dict, vocab:VOCAB=None)->dict:
    """上記 VOCAB を用いた下請け関数
    読み，音韻，モーラなどの情報を作成してデータセットといしての体裁を整える
    """
    _data = {}
    if vocab == None:
        vocab = VOCAB()
    x = [x[0] for x in _dict.values()]
    for _x in x:
        i = len(_data)  # 連番の番号を得る
        orth = vocab.ntt_orth2hira[_x] if _x in vocab.ntt_orth2hira else _x
        _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r = vocab.get7lists_from_orth(orth)
        phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r = vocab.get6ids(_phon, _orth, _yomi)
        _yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls = vocab.yomi2mora_transform(_yomi)    
        _data[i] = {'orig': orth, 'yomi': _yomi, 'ortho':_orth, 'ortho_ids': orth_ids, 
                    'ortho_r': _orth_r, 'ortho_ids_r': orth_ids_r, 
                    'phone':_phon, 'phone_ids': phon_ids, 'phone_r': _phon_r, 'phone_ids_r': phon_ids_r, 
                    'mora': _mora1, 'mora_r': _mora1_r, 'mora_ids': _mora_ids, 'mora_p': _mora_p, 
                    'mora_p_r': _mora_p_r, 'mora_p_ids': _mora_p_ids, 'mora_p_ids_r': _mora_p_ids_r,
                   }    

    return _data

# import torch
# import torch.utils.data
class Train_dataset(torch.utils.data.Dataset):
#class train_dataset(torch.utils.data.Dataset):
    """上で読み込んだ自作データ管理ライブラリ TLPA でも良いので冗長なのだが，訓練データセットと検証データセットとを明示的に定義"""
    
    #def __init__(self, tlpa=tlpa)->None:
    def __init__(self, 
                data:VOCAB=None,
                source_vocab:list=None,
                target_vocab:list=None,
                source_ids:str='mora_p_ids',                
                target_ids:str='mora_p_ids_r',
                ):
        #self.tlpa = tlpa
        #self.data = tlpa.train_data
        if data == None:
            self.data = VOCAB()
        else:
            self.data = data
        # self.order という名前の辞書を作成
        self.order = {i:self.data[x] for i, x in enumerate(self.data)}

        self.target_ids = target_ids
        self.source_ids = source_ids

        self.source_vocab = source_vocab if source_vocab != None else VOCAB().mora_p_vocab
        self.target_vocab = target_vocab if target_vocab != None else VOCAB().mora_p_vocab

        
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, x:int):
        return self.order[x][self.source_ids] + [self.source_vocab.index('<EOW>')], self.order[x][self.target_ids] + [self.target_vocab.index('<EOW>')]
    
    def convert_source_ids_to_tokens(self, ids:list):
        return [self.source_vocab[idx] for idx in ids]
    
    def convert_target_ids_to_tokens(self, ids:list):
        return [self.target_vocab[idx] for idx in ids]


#class _val_dataset(torch.utils.data.Dataset):
class Val_dataset(torch.utils.data.Dataset):
    """同じく検証データセットの定義"""

    #def __init__(self, data)->None:
    #def __init__(self, data=_dataset['SALA_R29']['pdata'])->None:
    def __init__(self,
                data:dict=None,
                source_vocab:list=None,
                target_vocab:list=None,
                source_ids:str='mora_p_ids',                
                target_ids:str='mora_p_ids_r',
                ):

        if 'pdata' in str(data.keys()):
            self.data = data['pdata']
        else:
            self.data = data

        self.order = {i:self.data[x] for i, x in enumerate(self.data)}

        self.target_ids = target_ids
        self.source_ids = source_ids

        self.source_vocab = source_vocab if source_vocab != None else VOCAB().mora_p_vocab
        self.target_vocab = target_vocab if target_vocab != None else VOCAB().mora_p_vocab

        
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, x:int):
        return self.order[x][self.source_ids] + [self.source_vocab.index('<EOW>')], self.order[x][self.target_ids] + [self.target_vocab.index('<EOW>')]
    
    def convert_source_ids_to_tokens(self, ids:list):
        return [self.source_vocab[idx] for idx in ids]
    
    def convert_target_ids_to_tokens(self, ids:list):
        return [self.target_vocab[idx] for idx in ids]


# import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
class EncoderRNN(nn.Module):
    """RNNによる符号化器"""
    def __init__(self, 
            n_inp:int=0, 
            n_hid:int=0,
            device=None):
        super().__init__()
        self.n_hid = n_hid if n_hid != 0 else 8
        self.n_inp = n_inp if n_inp != 0 else 8

        self.embedding = nn.Embedding(n_inp, n_hid)
        self.gru = nn.GRU(n_hid, n_hid)

    def forward(self, 
                inp:int=0, 
                hid:int=0):
        embedded = self.embedding(inp).view(1, 1, -1)
        out = embedded
        out, hid = self.gru(out, hid)
        return out, hid

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


class AttnDecoderRNN(nn.Module):
    """注意付き復号化器の定義"""
    def __init__(self, n_hid, n_out, dropout_p, max_length):
        super().__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.n_out, self.n_hid)
        self.attn = nn.Linear(self.n_hid * 2, self.max_length)
        self.attn_combine = nn.Linear(self.n_hid * 2, self.n_hid)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.n_hid, self.n_hid)
        self.out = nn.Linear(self.n_hid, self.n_out)

    def forward(self, inp, hid, encoder_outputs, device):
        embedded = self.embedding(inp).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hid[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        out = torch.cat((embedded[0], attn_applied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)

        out = F.relu(out)
        out, hid = self.gru(out, hid)

        out = F.log_softmax(self.out(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


def convert_ids2tensor(sentence_ids:list, device:torch.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """数値 ID リストをテンソルに変換
    例えば，[0,1,2] -> tensor([[0],[1],[2]])
    """
    return torch.tensor(sentence_ids, dtype=torch.long, device=device).view(-1, 1)


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion,
          max_length,
          target_vocab,
          teacher_forcing_ratio,
          device,
          )->float:
        #=tlpa.max_length):
    encoder_hidden = encoder.initHidden() # 符号化器の中間層を初期化
    encoder_optimizer.zero_grad()         # 符号化器の最適化関数の初期化
    decoder_optimizer.zero_grad()         # 復号化器の最適化関数の初期化

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.n_hid, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #print(f'ei:{ei} input_length:{input_length} encoder_output[0,0]:{encoder_output[0,0]}') # type(encoder_output):{type(encoder_output)}')
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[target_vocab.index('<SOW>')]], device=device)
    decoder_hidden = encoder_hidden
    

    ok_flag = True
    # 教師強制をするか否かを確率的に決める
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing: # 教師強制する場合 Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs,device=device)
            loss += criterion(decoder_output, target_tensor[di])
            ok_flag = (ok_flag) and (decoder_output.argmax() == target_tensor[di].detach().numpy()[0])
            decoder_input = target_tensor[di]  # Teacher forcing

    else: # 教師強制しない場合 Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs,device=device)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            #print(f'decoder_output:{decoder_output.argmax()}, target_tensor[di]:{target_tensor[di].detach().numpy()[0]}')
            #print(f'decoder_output==target_tesor[di].detach().numpy()[0]:{decoder_output.argmax()==target_tensor[di].detach().numpy()[0]}')
            ok_flag = (ok_flag) and (decoder_output.argmax() == target_tensor[di].detach().numpy()[0])
            if decoder_input.item() == target_vocab.index('<EOW>'):
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, ok_flag


def tokenize(inp_word:str, 
            _vocab:dict,
            max_length:int,
            pad:bool=False,
            )->list:
    if pad:
        ret = [_vocab.index('<PAD>') for _ in range(max_length - len(inp_word))]
    else:
        ret = []
    return ret + [_vocab.index(ch) if ch in _vocab else _vocab[_vocab.index('<UNK>')] for ch in inp_word]

#print(train_dataset.convert_source_ids_to_tokens(tokenize('watashi',pad=True)))


def calc_accuracy(
    _dataset, 
    encoder, 
    decoder, 
    max_length=None, 
    source_vocab=None,
    target_vocab=None,
    isPrint=False):

    ok_count = 0
    for i in range(_dataset.__len__()):
        _input_ids, _target_ids = _dataset.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(
            encoder=encoder, 
            decoder=decoder, 
            input_ids=_input_ids,
            max_length=max_length,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
        )
        ok_count += 1 if _target_ids == _output_ids else 0
        if (_target_ids != _output_ids) and (isPrint):
            print(i, _target_ids == _output_ids, _output_words, _input_ids, _target_ids)
            
    return ok_count/_dataset.__len__()


import time
import math

def asMinutes(s:int)->str:
    """時間変数を見やすいように，分と秒に変換して返す"""
    m = math.floor(s / 60)
    s -= m * 60
    return f'{int(m):2d}分 {int(s):2d}秒'
    return '%dm %ds' % (m, s)


def timeSince(since:time.time, 
            percent:time.time)->str:
    """開始時刻 since と，現在の処理が全処理中に示す割合 percent を与えて，経過時間と残り時間を計算して表示する"""
    now = time.time()  #現在時刻を取得
    s = now - since    # 開始時刻から現在までの経過時間を計算
    #s = since - now    
    es = s / (percent) # 経過時間を現在までの処理割合で割って終了予想時間を計算
    rs = es - s        # 終了予想時刻から経過した時間を引いて残り時間を計算

    return f'経過時間:{asMinutes(s)} (残り時間 {asMinutes(rs)})'


def check_vals_performance(encoder=None, decoder=None, 
                           _dataset=None, 
                           max_length=0, 
                           source_vocab=None,
                           target_vocab=None,
                           ):
    if _dataset == None or encoder == None or decoder == None or max_length == 0 or source_vocab == None:
        return
    print('検証データ:',end="")
    for _x in _dataset:
        ok_count = 0 
        for i in range(_dataset[_x].__len__()):
            _input_ids, _target_ids = _dataset[_x].__getitem__(i)
            _output_words, _output_ids, _attentions = evaluate(encoder, decoder, _input_ids, 
                                                               max_length, 
                                                               source_vocab=source_vocab,
                                                               target_vocab=target_vocab
                                                               )
            ok_count += 1 if _target_ids == _output_ids else 0
        print(f'{_x}:{ok_count/_dataset[_x].__len__():.3f},',end="")
    print()

import torch.nn as nn
import time

def fit(encoder:nn.Module, decoder:nn.Module, 
        epochs:int=1,
        lr:float=0.001,
        n_sample:int=3,
        teacher_forcing_ratio=False,
        train_dataset:torch.utils.data.Dataset=None,
        val_dataset:dict=None,
        source_vocab:list=None,
        target_vocab:list=None,
        params:dict=None,
        max_length:int=1,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
       )->list:
    
    start_time = time.time()
    
    encoder.train()
    decoder.train()
    encoder_optimizer = params['optim_func'](encoder.parameters(), lr=lr)
    decoder_optimizer = params['optim_func'](decoder.parameters(), lr=lr)
    criterion = params['loss_func']
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        
        ok_count = 0
        #エポックごとに学習順をシャッフルする
        learning_order = np.random.permutation(train_dataset.__len__())
        for i in range(train_dataset.__len__()):
            x = learning_order[i]   # ランダムにデータを取り出す 
            input_ids, target_ids = train_dataset.__getitem__(x)
            input_tensor = convert_ids2tensor(input_ids)
            target_tensor = convert_ids2tensor(target_ids)
            
            #訓練の実施
            loss, ok_flag = train(input_tensor, target_tensor, 
            #loss, ok_flag = lam.train(input_tensor, target_tensor, 
                                      encoder, decoder, 
                                      encoder_optimizer, decoder_optimizer, 
                                      criterion, 
                                      max_length=max_length,
                                      #max_length=_vocab.max_length,
                                      target_vocab=target_vocab,
                                      teacher_forcing_ratio=teacher_forcing_ratio,
                                      device=device)
            epoch_loss += loss
            ok_count += 1 if ok_flag else 0
        
        losses.append(epoch_loss/train_dataset.__len__())
        print(colored(f'エポック:{epoch:2d} 損失:{epoch_loss/train_dataset.__len__():.2f}', 'blue', attrs=['bold']),
              colored(f'{timeSince(start_time, (epoch+1) * train_dataset.__len__()/(epochs * train_dataset.__len__()))}', 
                      'cyan', attrs=['bold']),
              colored(f'訓練データの精度:{ok_count/train_dataset.__len__():.3f}', 'blue', attrs=['bold']))

        check_vals_performance(_dataset=val_dataset, 
        #lam.check_vals_performance(_dataset=val_dataset, 
                                   encoder=encoder, 
                                   decoder=decoder, 
                                   max_length=max_length,
                                   #max_length=_vocab.max_length,
                                   source_vocab=source_vocab,
                                   target_vocab=target_vocab)
        if n_sample > 0:
            evaluateRandomly(encoder, decoder, n=n_sample)
        
    return losses


def evaluate(encoder:nn.Module, 
             decoder:nn.Module, 
             input_ids,
             max_length,
             #max_length:int=_vocab.max_length,
             source_vocab,
             target_vocab,
            )->(list,torch.LongTensor):
    with torch.no_grad():
        input_tensor = convert_ids2tensor(input_ids)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.n_hid, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[source_vocab.index('<SOW>')]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words, decoded_ids = [], []  # decoded_ids を追加
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, device=device)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_ids.append(int(topi.squeeze().detach())) # decoded_ids に追加
            if topi.item() == target_vocab.index('<EOW>'):
                decoded_words.append('<EOW>')
                break
            else:
                decoded_words.append(target_vocab[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoded_ids, decoder_attentions[:di + 1]  # decoded_ids を返すように変更
        #return decoded_words, decoder_attentions[:di + 1]    

def evaluateRandomly(encoder:nn.Module, 
                     decoder:nn.Module, 
                     val_dataset=None,
                     n:int=5)->float:
    
    srcs, preds = [], []
    for x in np.random.randint(val_dataset.__len__(), size=n):
        input_ids, target_ids = val_dataset.__getitem__(x)
        input_words = val_dataset.convert_source_ids_to_tokens(input_ids)
        print(f'入力: {target_ids}<-{input_ids}:{input_words}')
        output_words, output_ids, attentions = evaluate(encoder, decoder, input_ids)

        srcs.append(input_words)
        preds.append(output_words)
        print(f'出力: {output_ids}',f':{output_words}')
        print('---')
    return srcs, preds

def _evaluateRandomly(encoder:nn.Module, 
                     decoder:nn.Module, 
                     n:int=5)->float:
    
    srcs, preds = [], []
    for x in np.random.randint(val_dataset.__len__(), size=n):
        input_ids, target_ids = val_dataset.__getitem__(x)
        input_words = val_dataset.convert_source_ids_to_tokens(input_ids)
        print(val_dataset.data[x])
        print(f'入力: {target_ids}<-{input_ids}:{input_words}')
        output_words, output_ids, attentions = evaluate(encoder, decoder, input_ids)

        srcs.append(input_words)
        preds.append(output_words)
        print(f'出力: {output_ids}',f':{output_words}')
        print('---')
    return srcs, preds

# -------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

def draw_sala_decay_results(sala_result:dict,
                            xlabel='崩壊率',
                            ylabel='精度',
                            title='デコーダ内の入力層から中間層への結合係数 "gru.weight_ih" の崩壊率',
                            save_fname='2022_0517lam_p2p_gru_weight_ih_sala.pdf'):

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)

    x = [k for k, v in sala_results.items()]
    y29hh = [v['sala_r29']['HH'] for k,v in sala_results.items()]
    y29hl = [v['sala_r29']['HL'] for k,v in sala_results.items()]
    y29lh = [v['sala_r29']['LH'] for k,v in sala_results.items()]
    y29ll = [v['sala_r29']['LL'] for k,v in sala_results.items()]
    y30hira2 = [v['sala_r30']['hira2'] for k,v in sala_results.items()]
    y30hira3 = [v['sala_r30']['hira3'] for k,v in sala_results.items()]
    y30hira4 = [v['sala_r30']['hira4'] for k,v in sala_results.items()]
    y30kata2 = [v['sala_r30']['kata2'] for k,v in sala_results.items()]
    y30kata3 = [v['sala_r30']['kata3'] for k,v in sala_results.items()]
    y30kata4 = [v['sala_r30']['kata4'] for k,v in sala_results.items()]
    y30kanj2 = [v['sala_r30']['kanj2'] for k,v in sala_results.items()]
    y30kanj3 = [v['sala_r30']['kanj3'] for k,v in sala_results.items()]
    y30kanj4 = [v['sala_r30']['kanj4'] for k,v in sala_results.items()]
    y31_2 = [v['sala_r31']['2'] for k,v in sala_results.items()]
    y31_3 = [v['sala_r31']['3'] for k,v in sala_results.items()]
    y31_4 = [v['sala_r31']['4'] for k,v in sala_results.items()]

    plt.plot(x, y29hh, c='blue', label='sala_r29:HH', marker='*', linewidth=3)
    plt.plot(x, y29hl, c='red', label='sala_r29:HL', marker='o', linewidth=3) # , linestyle='dashed')
    plt.plot(x, y29lh, c='cyan',label='sala_r29:LH', marker='<', linewidth=3)
    plt.plot(x, y29ll, c='green',label='sala_r29:LL', marker='>', linewidth=3)
    #plt.plot(x, y41, c='green', label='tlpa4非単語', marker='o', linewidth=3, linestyle='dashed')
    #plt.plot(x, y20, c='red', label='tlpa2単語', marker='*', linewidth=3)
    #plt.plot(x, y21, c='red', label='tlpa2非単語', marker='*', linewidth=3, linestyle='dashed')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title('中間表象とデコーダとの注意結合の崩壊率')
    #plt.title('デコーダの再帰結合の崩壊率')
    plt.title(title)
    plt.legend()
    if len(save_fname) > 0:
        fig.savefig(save_fname, dpi=100) 


def calc_val_tlpa(encoder,
                  decoder,
                  _data,
                  _cond,
                  source_vocab,
                  target_vocab,
                  max_length,
                  isPrint):
    n_hit = 0 
    n_word, c_word, n_non_word, c_non_word = 0, 0, 0, 0
    for i in range(_data.__len__()):
        _input_ids, _target_ids = _data.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(encoder, decoder, 
        #_output_words, _output_ids, _attentions = lam.evaluate(encoder, decoder, 
                                                               _input_ids, max_length=max_length, 
                                                               source_vocab=source_vocab,
                                                               target_vocab=target_vocab)
        _hit = _target_ids == _output_ids
        _cond_NW = len(_cond[f'{i+1:02d}'][1]) > 0
        #_cond_NW = len(_cond[f'{i+1:02d}'][1]) > 0
        if _hit:
            if _cond_NW:
                n_non_word += 1 
            else:
                n_word += 1
        if _cond_NW:
            c_word += 1
        else:
            c_non_word += 1

        n_hit += 1 if _hit else 0
        
    if not isPrint:
        return {'単語': n_word/c_word, '非単語': n_non_word/c_non_word } 
    else:
        print(f'単語:{n_word / c_word:.3f}',
              f'非単語:{n_non_word / c_non_word:.3f}')
        

def calc_val_tlpas(encoder,
                   decoder,
                   source_vocab,
                   target_vocab,
                   max_length,
                   _dataset,
                   X_vals,
                   isPrint):
    
    ret_dict = {'tlpa2':{}, 'tlpa3':{}, 'tlpa4':{}}

    ret_dict['tlpa2'] = calc_val_tlpa(encoder=encoder, decoder=decoder,
                                     _data=X_vals['tlpa2val'],
                                     #_data=X_vals['TLPA2'],
                                     _cond = _dataset['tlpa2']['rawdata'],
                                     source_vocab=source_vocab, target_vocab=target_vocab,
                                     max_length=max_length, isPrint=isPrint)
    ret_dict['tlpa3'] = calc_val_tlpa(encoder=encoder, decoder=decoder,
                                     _data=X_vals['tlpa3val'],
                                     #_data=X_vals['TLPA3'],
                                     _cond =_dataset['tlpa3']['rawdata'],
                                     source_vocab=source_vocab, target_vocab=target_vocab,
                                     max_length=max_length, isPrint=isPrint)
    ret_dict['tlpa4'] = calc_val_tlpa(encoder=encoder, decoder=decoder,
                                     _data=X_vals['tlpa4val'],
                                     #_data=X_vals['TLPA4'],
                                     _cond =_dataset['tlpa4']['rawdata'],
                                     source_vocab=source_vocab, target_vocab=target_vocab,
                                     max_length=max_length, isPrint=isPrint)

    # for _d in ret_dict.keys():
    #     if isPrint: print(f'{_d}', end=' ')
    #     ret_dict[_d] = calc_val_tlpa(encoder=encoder, decoder=decoder, 
    #                                  _data=eval(_d+'val'), cond=_dataset[_d],
    #                                  source_vocab=source_vocab, target_vocab=target_vocab,
    #                                  max_length=max_length, isPrint=isPrint)
    return ret_dict


# def _calc_val_sala_r29(encoder:nn.Module=None,
#                        decoder:nn.Module=None,
#                        _data:dict=None,
#                        _cond:dict=None,
#                       source_vocab:dict=None,
#                       target_vocab:dict=None,
#                       cond:dict=None,
#                       max_length:int=1):

#     ret_conds = {'HH':{}, 'HL':{}, 'LH':{}, 'LL':{}}
#     hits  = {'HH':0, 'HL':0, 'LH':0, 'LL':0}
#     total = {'HH':0, 'HL':0, 'LH':0, 'LL':0}
#     for i in range(_data.__len__()):
#         _input_ids, _target_ids = _data.__getitem__(i)
#         _output_words, _output_ids, _attentions = lam.evaluate(encoder, decoder, 
#                                                                _input_ids, max_length=max_length, 
#                                                                source_vocab=source_vocab,
#                                                                target_vocab=target_vocab)
#         _cond = cond[str(f'{i+1:02d}')][1]
#         if   _cond == '1000': __cond = 'HH'
#         elif _cond == '0100': __cond = 'HL'
#         elif _cond == '0010': __cond = 'LH'
#         elif _cond == '0001': __cond = 'LL'
#         if _target_ids == _output_ids: hits[__cond] += 1
#         total[__cond] += 1
            
#     return hits, total


def calc_val_sala_r29(encoder:nn.Module=None,
                      decoder:nn.Module=None,
                      _data:dict=None,
                      cond:dict=None,
                      source_vocab:dict=None,
                      target_vocab:dict=None,
                      #cond:dict=None,
                      max_length:int=1):

    ret_conds = {'HH':{}, 'HL':{}, 'LH':{}, 'LL':{}}
    hits  = {'HH':0, 'HL':0, 'LH':0, 'LL':0}
    total = {'HH':0, 'HL':0, 'LH':0, 'LL':0}
    for i in range(_data.__len__()):
        _input_ids, _target_ids = _data.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(encoder, decoder, _input_ids, max_length=max_length, 
                                                               source_vocab=source_vocab, target_vocab=target_vocab)
        _cond = cond[str(f'{i+1:02d}')][1]
        if   _cond == '1000': __cond = 'HH'
        elif _cond == '0100': __cond = 'HL'
        elif _cond == '0010': __cond = 'LH'
        elif _cond == '0001': __cond = 'LL'
        if _target_ids == _output_ids:
            hits[__cond] += 1
        total[__cond] += 1

    for _ret_cond in ret_conds.keys():
        ret_conds[_ret_cond] = hits[_ret_cond]/total[_ret_cond]
    return ret_conds    
    #return hits, total


def calc_val_sala_r30(encoder:nn.Module=None,
                      decoder:nn.Module=None,
                      _data:dict=None,
                      #_cond:dict=None,
                      source_vocab:dict=None,
                      target_vocab:dict=None,
                      cond:dict=None, 
                      max_length:int=1):
    
    ret_conds  = {'hira2':{}, 'hira3':{}, 'hira4':{}, 
                  'kata2':{}, 'kata3':{}, 'kata4':{}, 
                  'kanj2':{}, 'kanj3':{}, 'kanj4':{}}
    hits  = {'hira2':0, 'hira3':0, 'hira4':0, 'kata2':0, 'kata3':0, 'kata4':0, 'kanj2':0, 'kanj3':0, 'kanj4':0}
    total = {'hira2':0, 'hira3':0, 'hira4':0, 'kata2':0, 'kata3':0, 'kata4':0, 'kanj2':0, 'kanj3':0, 'kanj4':0}
    for i in range(_data.__len__()):
        _input_ids, _target_ids = _data.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(encoder, decoder, _input_ids, max_length=max_length, 
        #_output_words, _output_ids, _attentions = lam.evaluate(encoder, decoder, _input_ids, max_length=max_length, 
                                                               source_vocab=source_vocab, target_vocab=target_vocab)
        
        num = str(f'{i+1:02d}')
        _stim = cond[num][0]
        _cond_moji = cond[num][1]
        _cond_mora = cond[num][2]
        __cond = _cond_moji+_cond_mora
        _hit = _target_ids == _output_ids
        if _hit: hits[__cond] += 1
        total[__cond] += 1

    for _ret_cond in ret_conds.keys():
        ret_conds[_ret_cond] = hits[_ret_cond]/total[_ret_cond]
    return ret_conds    
    #return hits, total

# def _calc_val_sala_r30(encoder:nn.Module=None,
#                        decoder:nn.Module=None,
#                        _data:dict=None,
#                        _cond:dict=None,
#                       source_vocab:dict=None,
#                       target_vocab:dict=None,
#                       cond:dict=None,
#                       max_length:int=1):
    
#     hits  = {'hira2':0, 'hira3':0, 'hira4':0, 'kata2':0, 'kata3':0, 'kata4':0, 'kanj2':0, 'kanj3':0, 'kanj4':0}
#     total = {'hira2':0, 'hira3':0, 'hira4':0, 'kata2':0, 'kata3':0, 'kata4':0, 'kanj2':0, 'kanj3':0, 'kanj4':0}
#     for i in range(_data.__len__()):
#         _input_ids, _target_ids = _data.__getitem__(i)
#         _output_words, _output_ids, _attentions = lam.evaluate(encoder, decoder, _input_ids, max_length=max_length, 
#                                                                source_vocab=source_vocab, target_vocab=target_vocab)
        
#         num = str(f'{i+1:02d}')
#         _stim = cond[num][0]
#         _cond_moji = cond[num][1]
#         _cond_mora = cond[num][2]
#         __cond = _cond_moji+_cond_mora
#         _hit = _target_ids == _output_ids
#         if _hit: hits[__cond] += 1
#         total[__cond] += 1
        
#     return hits, total

def calc_val_sala_r31(encoder:nn.Module=None,
                      decoder:nn.Module=None,
                      _data:dict=None,
                      _cond:dict=None,
                      source_vocab:dict=None,
                      target_vocab:dict=None,
                      cond:dict=None,
                      max_length:int=1):
    
    hits  = {'2':0, '3':0, '4':0, '5':0}
    total  = {'2':0, '3':0, '4':0, '5':0}
    ret_conds  = {'2':{}, '3':{}, '4':{}}
    
    for i in range(_data.__len__()):
        _input_ids, _target_ids = _data.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(encoder, decoder, _input_ids, max_length=max_length, 
        #_output_words, _output_ids, _attentions = lam.evaluate(encoder, decoder, _input_ids, max_length=max_length, 
                                                               source_vocab=source_vocab, target_vocab=target_vocab)
        
        num = str(f'{i+1:02d}')
        __cond = cond[num][1]
        _hit = _target_ids == _output_ids
        if _hit: hits[__cond] += 1
        total[__cond] += 1
        
    for _ret_cond in ret_conds.keys():
        ret_conds[_ret_cond] = hits[_ret_cond]/total[_ret_cond]
    return ret_conds    
    #return hits, total

# def _calc_val_sala_r31(encoder:nn.Module=None,
#                        decoder:nn.Module=None,
#                       _data:dict=None,
#                       _cond:dict=None,
#                       source_vocab:dict=None, 
#                       target_vocab:dict=None,
#                       cond:dict=None,
#                       max_length:int=1):
    
#     hits  = {'2':0, '3':0, '4':0, '5':0}
#     total  = {'2':0, '3':0, '4':0, '5':0}
#     for i in range(_data.__len__()):
#         _input_ids, _target_ids = _data.__getitem__(i)
#         _output_words, _output_ids, _attentions = lam.evaluate(encoder, decoder, _input_ids, max_length=max_length, 
#                                                                source_vocab=source_vocab, target_vocab=target_vocab)
        
#         num = str(f'{i+1:02d}')
#         __cond = cond[num][1]
#         _hit = _target_ids == _output_ids
#         if _hit: hits[__cond] += 1
#         total[__cond] += 1
#     return hits, total


def calc_val_salas(encoder:nn.Module=None,
                   decoder:nn.Module=None,
                   source_vocab:dict=None,
                   target_vocab:dict=None,
                   max_length:int=1,
                   X_vals:dict=None,
                   _dataset:dict=None,
                   isPrint=False):
    
    ret_results = {'sala_r29':{}, 'sala_r30':{}, 'sala_r31':{}}
    
    __x            = calc_val_sala_r29(encoder=encoder, decoder=decoder, 
                                       source_vocab=source_vocab, target_vocab=target_vocab, 
                                       _data=X_vals['sala_r29val'],
                                       #_data=X_vals['SALA_R29'],
                                       cond=_dataset['sala_r29']['Cond'],
                                       max_length=max_length)    
    #_hits, _total = _calc_val_sala_r29(encoder=encoder, decoder=decoder, 
    #                                   source_vocab=source_vocab, target_vocab=target_vocab, 
    #                                   max_length=_vocab.max_length)
    ret_results['sala_r29'] = __x
    #ret_results['sala_r29'] = {'_hits':_hits, '_total':_total}
    if isPrint: 
        print('sala_r29', end=" ")
        print(__x)
        # for h,t in zip(_hits,_total):
        #     print(f'{h}:{_hits[h]/_total[t]:.3f}', end=" ")
        # print()

    __x            = calc_val_sala_r30(encoder=encoder,decoder=decoder, 
                                       source_vocab=source_vocab, target_vocab=target_vocab, 
                                       _data=X_vals['sala_r30val'],
                                       #_data=X_vals['SALA_R30'],
                                       cond=_dataset['sala_r30']['Cond'],
                                       max_length=max_length)
    #_hits, _total = calc_val_sala_r30(encoder=encoder,decoder=decoder, 
    #                                  source_vocab=source_vocab, target_vocab=target_vocab, 
    #                                  max_length=_vocab.max_length)

    ret_results['sala_r30'] = __x
    #ret_results['sala_r30'] = {'_hits':_hits, '_total':_total}
    if isPrint:
        print('sala_r30', end=" ")
        for h,t in zip(_hits,_total):
            print(f'{h}:{_hits[h]/_total[t]:.3f}', end=" ")
        print()

    __x            = calc_val_sala_r31(encoder=encoder,decoder=decoder, source_vocab=source_vocab, 
                                       target_vocab=target_vocab, max_length=max_length,
                                       _data=X_vals['sala_r31val'],
                                       #_data=X_vals['SALA_R31'],
                                       cond=_dataset['sala_r31']['Cond'],
                                       )
    #_hits, _total = calc_val_sala_r31(encoder=encoder,decoder=decoder, source_vocab=source_vocab, 
    #                                  target_vocab=target_vocab, max_length=_vocab.max_length)
    ret_results['sala_r31'] = __x
    #ret_results['sala_r31'] = {'_hits':_hits, '_total':_total}
    if isPrint:
        print('sala_r31', end=" ")
        for h,t in zip(_hits,_total):
            print(f'{h}:{_hits[h]/_total[t]:.3f}', end=" ")
        print()
    else:
        return ret_results



def get_soure_and_target_from_params(params=None,
                                    _vocab=None,
                                    source=None,
                                    target=None,
                                     is_print:bool=True):
    #source = params['source']
    #target = params['target']

    if source == 'orthography':
        source_vocab = _vocab.ortho_vocab
        source_ids = 'orth_ids'
    elif source == 'phonology':
        source_vocab = _vocab.phone_vocab
        source_ids = 'phone_ids'
    elif source == 'mora':
        source_vocab = _vocab.mora_vocab
        source_ids = 'mora_ids'
    elif source == 'mora_p':
        source_vocab = _vocab.mora_p_vocab
        source_ids = 'mora_p_ids'
    elif source == 'mora_p_r':
        source_vocab = _vocab.mora_p_vocab
        source_ids = 'mora_p_ids_r'

    
    if target == 'orthography':
        target_vocab = _vocab.ortho_vocab
        target_ids = 'ortho_ids'
    elif target == 'phonology':
        target_vocab = _vocab.phone_vocab
        target_ids = 'phone_ids'
    elif target == 'mora':
        target_vocab = _vocab.mora_vocab
        target_ids = 'mora_ids'
    elif target == 'mora_p':
        target_vocab = _vocab.mora_p_vocab
        target_ids = 'mora_p_ids'
    elif target == 'mora_p_r':
        target_vocab = _vocab.mora_p_vocab
        target_ids = 'mora_p_ids_r'
    

    if is_print:
        print(colored(f'source:{source}','blue', attrs=['bold']), f'{source_vocab}')
        print(colored(f'target:{target}','cyan', attrs=['bold']), f'{target_vocab}')
        print(colored(f'source_ids:{source_ids}','blue', attrs=['bold']), f'{source_ids}')
        print(colored(f'target_ids:{target_ids}','cyan', attrs=['bold']), f'{target_ids}')

    return source_vocab, source_ids, target_vocab, target_ids


def fix_seed(seed:int):
    random.seed(seed)  # for random
    np.random.seed(seed) # for numpy

    # for pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def make_X_vals(_dataset=None,
                source_vocab=None,
                target_vocab=None,
                ):

    if _dataset == None:
        print('_dataset must be set')
        sys.exit()

    sala_r29val = Val_dataset(data=_dataset['sala_r29'],
                            source_vocab=source_vocab, 
                            target_vocab=target_vocab)

    sala_r30val = Val_dataset(data=_dataset['sala_r30'],
                            source_vocab=source_vocab, 
                            target_vocab=target_vocab)

    sala_r31val = Val_dataset(data=_dataset['sala_r31'],
                            source_vocab=source_vocab, 
                            target_vocab=target_vocab)

    tlpa2val    = Val_dataset(data=_dataset['tlpa2'],
                            source_vocab=source_vocab, 
                            target_vocab=target_vocab)

    tlpa3val    = Val_dataset(data=_dataset['tlpa3'],
                            source_vocab=source_vocab, 
                            target_vocab=target_vocab)

    tlpa4val    = Val_dataset(data=_dataset['tlpa4'],
                            source_vocab=source_vocab, 
                            target_vocab=target_vocab)

    X_vals = { 'SALA_R29': sala_r29val,'SALA_R30': sala_r30val,'SALA_R31': sala_r31val,
                'TLPA2': tlpa2val,'TLPA3': tlpa3val,'TLPA4': tlpa4val}

    X_vals = { 'sala_r29val': sala_r29val,'sala_r30val': sala_r30val,'sala_r31val': sala_r31val,
                'tlpa2val': tlpa2val, 'tlpa3val': tlpa3val, 'tlpa4val': tlpa4val}


    return X_vals

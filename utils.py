from config.pattern import *
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from underthesea import word_tokenize
from collections import Counter
from torchtext.vocab import Vocab
import torch
import joblib
from tqdm import tqdm
import html
import os
from torch.nn.utils.rnn import pad_sequence

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def vi_tokenizer(text: str):
    return word_tokenize(text)


def preprocess(text: str):
    text = html.unescape(text)
    text = BREAK_PATTERN.sub(' ', text)
    # repr to change string to raw string
    text = EMOJI_PATTERN.sub(' ', repr(text)[1:-1])
    text = WEB_PATTERN.sub('web_add', text)
    text = NUM_PATTERN.sub('num_', text)
    text = re.sub(r'-', ' ', text)
    text = SPACES_PATTERN.sub(' ', text).strip()
    return text


class ReadData():
    '''
    data: (src, trg)
    '''

    def __init__(self, max_len, srcpath, trgpath):
        self.srcpath = srcpath
        self.trgpath = trgpath
        self.max_len = max_len
        self.data = []

    def read_data(self, src_tokenizer=en_tokenizer, trg_tokenizer=vi_tokenizer):
        raw_src_iter = iter(open(self.srcpath, 'r', encoding="utf8"))
        raw_trg_iter = iter(open(self.trgpath, 'r', encoding="utf8"))
        for (raw_src, raw_trg) in tqdm(zip(raw_src_iter, raw_trg_iter)):
            raw_src = src_tokenizer(preprocess(raw_src.rstrip("\n")))
            raw_trg = trg_tokenizer(preprocess(raw_trg.rstrip("\n")))
            if (len(raw_src) <= self.max_len) & (len(raw_trg) <= self.max_len):
                self.data.append((raw_src, raw_trg))
        raw_src_iter.close()
        raw_trg_iter.close()

    def process(self, src_vocab, trg_vocab):
        processed = []
        for (raw_src, raw_trg) in tqdm(self.data):
            src_tensor_ = torch.tensor([src_vocab[token] for token in raw_src],
                                       dtype=torch.long)
            trg_tensor_ = torch.tensor([trg_vocab[token] for token in raw_trg],
                                       dtype=torch.long)
            processed.append((src_tensor_, trg_tensor_))
        return processed


class ReadVocab():
    def __init__(self, vocab_size, srcname, trgname):
        self.counter_src = Counter()
        self.counter_trg = Counter()
        self.vocab_src = Vocab(self.counter_src)
        self.vocab_trg = Vocab(self.counter_trg)
        self.vocab_size = vocab_size
        self.specials_token = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.name = (srcname, trgname)

    def build_vocab(self, data):
        for ite in data:
            self.counter_src.update(ite[0])
            self.counter_trg.update(ite[1])
        if sum(self.counter_src.values()) < self.vocab_size:
            self.vocab_src = Vocab(
                self.counter_src, max_size=None,specials=self.specials_token)
        else:
            self.vocab_src = Vocab(
                self.counter_src, self.vocab_size, specials=self.specials_token)
        if sum(self.counter_trg.values()) < self.vocab_size:
            self.vocab_trg = Vocab(
                self.counter_trg, max_size=None, specials=self.specials_token)
        else:
            self.vocab_trg = Vocab(
                self.counter_trg, self.vocab_size, specials=self.specials_token)

    def save_vocab(self, src_vocab_dir, trg_vocab_dir):
        joblib.dump(self.vocab_src.__getstate__(), src_vocab_dir)
        joblib.dump(self.vocab_trg.__getstate__(), trg_vocab_dir)

    def load_vocab(self, src_vocab_dir, trg_vocab_dir):
        self.vocab_src.__setstate__(joblib.load(src_vocab_dir))
        self.vocab_trg.__setstate__(joblib.load(trg_vocab_dir))


class ProcessData():
    def __init__(self, lazy_load):
        self.src_vocab_size = None
        self.trg_vocab_size = None
        self.data_prc = None
        self.data_iter = None
        self.vocab_src = None
        self.vocab_trg = None
        self.lazy_load = lazy_load

    def process(self, max_len, srcpath, trgpath, vocab_size, batch_size, srcname, trgname, src_vocab_dir, trg_vocab_dir, train=True, save_dir=None):
        print("[!] Preparing dataset...")
        if self.lazy_load:
            self.data_prc = joblib.load(save_dir)
            vocabinit = ReadVocab(vocab_size, srcname, trgname)
            vocabinit.load_vocab(src_vocab_dir, trg_vocab_dir)
        else:
            datainit = ReadData(max_len, srcpath, trgpath)
            datainit.read_data()
            vocabinit = ReadVocab(vocab_size, srcname, trgname)
            if train:
                vocabinit.build_vocab(datainit.data)
                print("[Saving] Vocabulary...")
                if not os.path.isdir("data/vocab"):
                    os.makedirs("data/vocab")
                vocabinit.save_vocab(src_vocab_dir, trg_vocab_dir)
             
            else:
                vocabinit.load_vocab(src_vocab_dir, trg_vocab_dir)
            self.vocab_src = vocabinit.vocab_src
            self.vocab_trg = vocabinit.vocab_trg
            self.src_vocab_size, self.trg_vocab_size = len(
                self.vocab_src), len(self.vocab_trg)
            print("[%s_vocab]:%d [%s_vocab]:%d" % (vocabinit.name[0],
                                                   self.src_vocab_size, vocabinit.name[1], self.trg_vocab_size))                   
            self.data_prc = datainit.process(
                vocabinit.vocab_src, vocabinit.vocab_trg)   
            
            if save_dir:
                print("[Saving] Processed data...")
                if not os.path.isdir("data/processed"):
                    os.makedirs("data/processed")
                joblib.dump(self.data_prc, save_dir)
                
        self.vocab_src = vocabinit.vocab_src
        self.vocab_trg = vocabinit.vocab_trg
        self.src_vocab_size, self.trg_vocab_size = len(
            self.vocab_src), len(self.vocab_trg)
        print("[%s_vocab]:%d [%s_vocab]:%d" % (vocabinit.name[0],
                                               self.src_vocab_size, vocabinit.name[1], self.trg_vocab_size))   
        self.data_iter = DataLoader(self.data_prc, batch_size=batch_size,
                                    shuffle=True, collate_fn=self.generate_batch)

    def generate_batch(self, data_batch):
        src_batch, trg_batch = [], []
        for (src_item, trg_item) in data_batch:
            src_batch.append(
                torch.cat([torch.tensor([self.vocab_src['<sos>']]), src_item, torch.tensor([self.vocab_src['<eos>']])], dim=0))
            trg_batch.append(
                torch.cat([torch.tensor([self.vocab_trg['<sos>']]), trg_item, torch.tensor([self.vocab_trg['<eos>']])], dim=0))
        src_batch = pad_sequence(
            src_batch, padding_value=self.vocab_src['<pad>'])
        trg_batch = pad_sequence(
            trg_batch, padding_value=self.vocab_trg['<pad>'])
        return src_batch, trg_batch

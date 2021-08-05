import torch
from torch import nn
from torch import optim

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import time
# from reporter import Reporter
import os
from tqdm import tqdm
from torch.nn import Transformer
from model.transformer.Layers import EncoderLayer, DecoderLayer
from model.transformer.Sublayers import Norm
from model.transformer.Batch import create_masks
from model.transformer.Embed import Embedder, PositionalEncoder
import copy

SOS_token = 2
EOS_token = 3
MAX_LENGTH = 100
PAD_IDX = 1


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def get_model(opt, src_vocab, trg_vocab, device, path=None):

    assert opt.embed_size % opt.n_head == 0
    assert opt.dropout_encoder < 1

    model = Transformer(src_vocab, trg_vocab,
                        opt.embed_size, opt.N, opt.n_head)

    if path is not None:
        print("loading pretrained models...")
        model.load_state_dict(torch.load(
            path, map_location=device)['model_state_dict'])
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model.to(device)


class MNTModel():
    def __init__(self, config, src_vocab_size, trg_vocab_size, trg_vocab, device, grad_clip, patience, min_delta):
        print("[!] Instantiating models...")
        self.trg_size = trg_vocab_size
        self.trg_vocab = trg_vocab

        self.model = get_model(config, src_vocab_size, trg_vocab_size, device)
        self.optimizer = optim.Adam(self.model.parameters(
        ), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        print(self.model)

        self.best_val_loss = None
        self.grad_clip = grad_clip
        self.early_stopping = EarlyStopping(patience, min_delta)
        self.device = device
        self.config = config

    def train_epoch(self, n_epochs, train_iter, val_iter, save_model_path):
        # reporter = Reporter("logs", "mnt_envi")
        for e in range(1, n_epochs+1):
            start_time = time.time()
            train_loss = self.train(e, train_iter)
            end_time = time.time()
            val_loss = self.evaluate(val_iter)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                break
            print((f"Epoch: {e}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                   f"Epoch time = {(end_time - start_time):.3f}s"))
            # reporter.log_metric("val_loss", float(val_loss), e)

            # Save the model if the validation loss is the best we've seen so far.
            if not self.best_val_loss or val_loss < self.best_val_loss:
                print("[!] saving model...")
                if not os.path.isdir("save"):
                    os.makedirs("save")
                torch.save({'model_state_dict': self.model.state_dict(),
                            'epoch': e,
                            'loss': '%5.3f' % val_loss},
                           save_model_path)
                self.best_val_loss = val_loss

    def train(self, e, train_iter):
        self.model.train()
        total_loss = 0
        pad = self.trg_vocab.stoi['<pad>']
        with tqdm(train_iter, unit="batch") as tepoch:
            for src, trg in tepoch:
                tepoch.set_description(f"Epoch {e}")

                src, trg_ = src.transpose(0, 1).to(
                    self.device), trg.transpose(0, 1).to(self.device)  # TxB
                trg = trg_[:, :-1]
                src_mask, trg_mask = create_masks(src, trg, self.device)
                self.optimizer.zero_grad()
                output = self.model(src, trg, src_mask, trg_mask)
                loss = F.cross_entropy(output.view(-1, output.size(-1)),
                                       trg_[:, 1:].contiguous().view(-1), ignore_index=PAD_IDX)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                tepoch.set_postfix(loss=loss.item())

                total_loss += loss.item()
        return total_loss/len(train_iter)

    def evaluate(self, val_iter):
        with torch.no_grad():
            self.model.eval()
            pad = self.trg_vocab.stoi['<pad>']
            total_loss = 0
            for idx, (src, trg) in enumerate(val_iter):
                src, trg_ = src.transpose(0, 1).to(
                    self.device), trg.transpose(0, 1).to(self.device)
                trg = trg_[:, :-1]
                src_mask, trg_mask = create_masks(src, trg, self.device)
                output = self.model(src, trg, src_mask, trg_mask)
                loss = F.cross_entropy(output.view(-1, output.size(-1)),
                                       trg_[:, 1:].contiguous().view(-1), ignore_index=PAD_IDX)
                total_loss += loss.data.item()
            return total_loss / len(val_iter)

    def eval(self):
        self.model.eval()

    def lazyload(self, path, device):
        self.model.load_state_dict(torch.load(
            path, map_location=device)['model_state_dict'])


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience, min_delta):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

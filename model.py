import math
import torch
import random
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time
from reporter import Reporter
import os
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers, dropout):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg):
        try:
            max_len = trg.size(0)

        except:
            max_len = trg
            self.teacher_forcing_ratio = 1

        batch_size = src.size(1)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).to(
            self.device)  # T*B*V

        encoder_output, hidden = self.encoder(src)  # T*B*H
        hidden = hidden[:self.decoder.n_layers]
        try:
            output = Variable(trg.data[0, :])  # sos # B
        except:
            output = Variable(torch.tensor(np.array([2]))).to(
                self.device)  # vi_vocab.stoi['<sos>']

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output
            if self.teacher_forcing_ratio == 1:
                is_teacher = False
            else:
                is_teacher = random.random() < self.teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(
                trg.data[t] if is_teacher else top1).to(self.device)
        return outputs


class MNTModel():
    def __init__(self, src_size, trg_size, trg_vocab, embed_size, hidden_size, n_layers_encoder, n_layers_decoder, dropout_encoder, dropout_decoder, device, teacher_forcing_ratio, learning_rate, grad_clip, patience, min_delta):
        print("[!] Instantiating models...")
        self.trg_size = trg_size
        self.trg_vocab = trg_vocab
        self.encoder = Encoder(src_size, embed_size, hidden_size,
                               n_layers=n_layers_encoder, dropout=dropout_encoder)
        self.decoder = Decoder(embed_size, hidden_size, self.trg_size,
                               n_layers=n_layers_decoder, dropout=dropout_decoder)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder, device,
                               teacher_forcing_ratio).to(device)
        self.optimizer = optim.Adam(
            self.seq2seq.parameters(), lr=learning_rate)
        print(self.seq2seq)
        self.best_val_loss = None
        self.grad_clip = grad_clip
        self.early_stopping = EarlyStopping(patience, min_delta)
        self.device = device

    def train_epoch(self, n_epochs, train_iter, val_iter, save_model_path):
        reporter = Reporter("logs", "mnt_envi")
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
            reporter.log_metric("val_loss", float(val_loss), e)

            # Save the model if the validation loss is the best we've seen so far.
            if not self.best_val_loss or val_loss < self.best_val_loss:
                print("[!] saving model...")
                if not os.path.isdir("save"):
                    os.makedirs("save")
                torch.save({'model_state_dict': self.seq2seq.state_dict(),
                            'epoch': e,
                            'loss': '%5.3f' % val_loss},
                           save_model_path)
                self.best_val_loss = val_loss

    def train(self, e, train_iter):
        self.seq2seq.train()
        total_loss = 0
        pad = self.trg_vocab.stoi['<pad>']
        with tqdm(train_iter, unit="batch") as tepoch:
            for src, trg in tepoch:
                tepoch.set_description(f"Epoch {e}")

                src, trg = src.to(self.device), trg.to(self.device)
                self.optimizer.zero_grad()
                output = self.seq2seq(src, trg)
                loss = F.nll_loss(output[1:].view(-1, self.trg_size),
                                  trg[1:].contiguous().view(-1),
                                  ignore_index=pad)
                loss.backward()
                clip_grad_norm_(self.seq2seq.parameters(), self.grad_clip)
                self.optimizer.step()

                tepoch.set_postfix(loss=loss.item())

                total_loss += loss.item()
        return total_loss/len(train_iter)

    def evaluate(self, val_iter):
        with torch.no_grad():
            self.seq2seq.eval()
            pad = self.trg_vocab.stoi['<pad>']
            total_loss = 0
            for idx, (src, trg) in enumerate(val_iter):
                src = src.data.to(self.device)
                trg = trg.data.to(self.device)
                output = self.seq2seq(src, trg)
                loss = F.nll_loss(output[1:].view(-1, self.trg_size),
                                  trg[1:].contiguous().view(-1),
                                  ignore_index=pad)
                total_loss += loss.data.item()
            return total_loss / len(val_iter)

    def lazyload(self, path):
        self.seq2seq.load_state_dict(torch.load(path)['model_state_dict'])


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

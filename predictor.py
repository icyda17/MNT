from model import *
from utils import *
from translate import translate_beam_search


class Predictor():
    def __init__(self, device):

        self.device = device
        cfg = ReadConfig()
        # load vocabulary
        self.vocab = ReadVocab(cfg.vocab_size, "EN", "VI")
        self.vocab.load_vocab(cfg.src_vocab_dir, cfg.trg_vocab_dir)
        self.MNT = MNTModel(len(self.vocab.vocab_src), len(self.vocab.vocab_trg), self.vocab.vocab_trg, cfg.embed_size,
                            cfg.hidden_size, cfg.n_layers_encoder, cfg.n_layers_decoder,
                            cfg.dropout_encoder, cfg.dropout_decoder, self.device, cfg.teacher_forcing_ratio,
                            cfg.learning_rate, cfg.grad_clip, cfg.patience, cfg.min_delta)
        self.MNT.lazyload(cfg.path_model)
        # self.MNT.seq2seq.eval()
        print("Load model!")
        self.beamsearch = cfg.beamsearch
        self.max_len = cfg.max_len

    def string2idx(self, text):
        '''convert text to list of indexes'''
        prep = en_tokenizer(preprocess(text))
        assert len(prep) <= self.max_len, "Exceed max length: %s" % self.max_len
        prep = [self.vocab.vocab_src['<sos>']]+[self.vocab.vocab_src[token]
                                                for token in prep]+[self.vocab.vocab_src['<eos>']]
        post = torch.tensor(prep, dtype=torch.long)
        post = pad_sequence(
            [post], padding_value=self.vocab.vocab_src['<pad>'])
        return post  # T*B(1)

    def decode(self, ids):
        decoded_w = []
        for t in ids[0]:
            decoded_w.append(self.vocab.vocab_trg.itos[t])
            if t == 3:
                break

        translate = ' '.join(decoded_w)
        return translate

    def predict(self, text):
        text = self.string2idx(text).to(self.device)
        if self.beamsearch:
            sent = translate_beam_search(text, self.model)
            s = sent
        else:
            s = self.MNT.seq2seq(text, self.max_len)
            s = s.squeeze(1).cpu().detach().numpy().tolist()

        s = self.decode(s)

        return s

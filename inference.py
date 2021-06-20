from utils import *
from model import *



class Inference():

    def __init__(self, device, max_len):
        self.MNT = None
        self.device = device
        self.vocab = None
        self.max_len = max_len

    def load(self, vocab_size, embed_size, hidden_size, n_layers_encoder, n_layers_decoder, dropout_encoder, dropout_decoder, device, teacher_forcing_ratio, learning_rate, grad_clip, patience, min_delta, path_model, src_vocab_dir, trg_vocab_dir):
        # load vocabulary
        self.vocab = ReadVocab(vocab_size, "EN", "VI")
        self.vocab.load_vocab(src_vocab_dir, trg_vocab_dir)
        self.MNT = MNTModel(len(self.vocab.vocab_src), len(self.vocab.vocab_trg), self.vocab.vocab_trg, embed_size, hidden_size, n_layers_encoder, n_layers_decoder,
                            dropout_encoder, dropout_decoder, self.device, teacher_forcing_ratio, learning_rate, grad_clip, patience, min_delta)
        self.MNT.lazyload(path_model)
        self.MNT.seq2seq.eval()
        print("Load model!")

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

    def infer(self, text, beam_width, alpha):
        INPUT = self.string2idx(text).to(self.device)
        OUTPUT = self.MNT.seq2seq(INPUT, self.max_len)
        bs = BeamSearch(beam_width, alpha)
        bs.execute(OUTPUT)
        bs.refine()

        decoded_w = []
        for t in bs.out[0]:
            decoded_w.append(self.vocab.vocab_trg.itos[t])
            if t == 3:
                break

        translate = ' '.join(decoded_w)
        return translate


class BeamSearch():
    def __init__(self, beam_width, alpha):
        self.k = beam_width
        self.sequences = [[list(), 0.0]]
        self.alpha = alpha
        self.out = None
    def execute(self, data):
        data = data.squeeze(1).cpu().detach().numpy()
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(self.sequences)):
                seq, score = self.sequences[i]
                # instead of exploring all the labels, explore only k best at the current time
                # select k best
                best_k = np.argsort(row)[-self.k:]
                # explore k best
                for j in best_k:
                    candidate = [seq + [j], score + row[j]]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(
                all_candidates, key=lambda tup: tup[1], reverse=True)
            # select k best
            self.sequences = ordered[:self.k]
    def refine(self):
        for i in range(len(self.sequences)):
            ty=0
            for j in self.sequences[i][0]:
                ty+=1
                if j == 3:
                    self.sequences[i][1] /= (ty**self.alpha)
        self.out = sorted(
                self.sequences, key=lambda tup: tup[1], reverse=True)[0]

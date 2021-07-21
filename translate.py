import torch
from model import Beam
from torch.nn.functional import log_softmax

def translate_beam_search(text, model, beam_size, candidates, max_seq_length, sos_token, eos_token):
    # text: T*B(1)
    model.eval()
    device = text.device

    with torch.no_grad():
        encoder_output, memory = model.forward_encoder(text)  # T*B*H: Memory: hidden layer
        sent = beamsearch(memory, encoder_output, model, device, beam_size,
                          candidates, max_seq_length, sos_token, eos_token)

    return sent


def beamsearch(memory, encoder_output, model, device, beam_size=4, candidates=1, max_seq_length=100, sos_token=2, eos_token=3):
    # encoder_output T*B*H
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates,
                ranker=None, start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
        memory = model.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs, memory = model.forward_decoder(tgt_inp, memory, encoder_output)

            log_prob = log_softmax(
                decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]

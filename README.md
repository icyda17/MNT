# mini seq2seq
Minimal Seq2Seq model with attention for neural machine translation in PyTorch.

This implementation focuses on the following features:

- Modular structure to be used in other projects
- Minimal code for readability
- Full utilization of batches and GPU.
- Provide beam-search

This implementation relies on [torchtext](https://github.com/pytorch/text) to minimize dataset management and preprocessing parts.

## Model description

* Encoder: Bidirectional GRU
* Decoder: GRU with Attention Mechanism
* Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

![](http://www.wildml.com/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM.png)

## How to run

* conda create new environment refer to [file](requirements.txt)

* Download tokenizers of English:
```
python -m spacy download en_core_web_sm
```
* Remember to modify [config file](config/config_mdl_sml.yaml) before running
```
CUDA_VISIBLE_DEVICES=1 python train.py
```
## References

Based on the following implementations

* [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [@spro/practical-pytorch](https://github.com/spro/practical-pytorch)
* [@AuCson/PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)

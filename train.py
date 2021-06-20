import os
import yaml
from utils import *
import torch
import pprint
from model import *


# tensorboard --logdir logs --port 6007 --host 0.0.0.0


def main():
    torch.manual_seed(0)

    path_cur = os.path.dirname(os.path.abspath(__file__))  # project path
    with open(os.path.join(path_cur, "config/config_model.yaml")) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(config)

    params_ = config['params']
    hyperparams_ = config['hyperparams']
    model_ = config['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_vocab_dir, trg_vocab_dir = config['data']['vocab']
    train_processed_dir, valid_processed_dir, test_processed_dir = config['data']['processed']

    trainsrc_path, valsrc_path, testsrc_path = [os.path.join(
        path_cur, sub_path) for sub_path in config['data']['src']]
    traintrg_path, valtrg_path, testtrg_path = [os.path.join(
        path_cur, sub_path) for sub_path in config['data']['trg']]

    if (os.path.isfile(os.path.join(path_cur, src_vocab_dir))) and (os.path.isfile(os.path.join(path_cur, trg_vocab_dir))) and (os.path.isfile(os.path.join(path_cur, train_processed_dir))) and (os.path.isfile(os.path.join(path_cur, valid_processed_dir))) and (os.path.isfile(os.path.join(path_cur, test_processed_dir))):
        lazy_load = True
    else:
        lazy_load = False

    trainprocess = ProcessData(lazy_load)
    validprocess = ProcessData(lazy_load)
    testprocess = ProcessData(lazy_load)

    trainprocess.process(max_len=params_['max_len'], srcpath=trainsrc_path, trgpath=traintrg_path,
                         vocab_size=params_['vocab_size'], batch_size=hyperparams_['batch_size'], srcname='EN', trgname='VI',
                         src_vocab_dir=src_vocab_dir, trg_vocab_dir=trg_vocab_dir, train=True, save_dir=train_processed_dir)
    validprocess.process(max_len=params_['max_len'], srcpath=valsrc_path, trgpath=valtrg_path,
                         vocab_size=params_['vocab_size'], batch_size=hyperparams_['batch_size'], srcname='EN', trgname='VI',
                         src_vocab_dir=src_vocab_dir, trg_vocab_dir=trg_vocab_dir, train=False, save_dir=valid_processed_dir)
    testprocess.process(max_len=params_['max_len'], srcpath=testsrc_path, trgpath=testtrg_path,
                        vocab_size=params_['vocab_size'], batch_size=hyperparams_['batch_size'], srcname='EN', trgname='VI',
                        src_vocab_dir=src_vocab_dir, trg_vocab_dir=trg_vocab_dir, train=False, save_dir=test_processed_dir)

    print("[TRAIN]:%d [TRAIN-ITER]: %d \t[VALID]:%d [VALID-ITER]: %d \t[TEST]:%d [TEST-ITER]: %d"
          % (len(trainprocess.data_prc), len(trainprocess.data_iter), len(validprocess.data_prc),
             len(validprocess.data_iter), len(testprocess.data_prc), len(testprocess.data_iter)))

    model = MNTModel(src_size=trainprocess.src_vocab_size, trg_size=trainprocess.trg_vocab_size,
                     trg_vocab=trainprocess.vocab_trg, embed_size=params_[
                         'embed_size'],
                     hidden_size=params_['hidden_size'], n_layers_encoder=model_[
                         'encoder']['n_layers'],
                     n_layers_decoder=model_['decoder']['n_layers'], dropout_encoder=model_[
                         'encoder']['dropout'],
                     dropout_decoder=model_[
                         'encoder']['dropout'], device=device,
                     teacher_forcing_ratio=model_[
                         'seq2seq']['teacher_forcing_ratio'],
                     learning_rate=hyperparams_[
                         'learning_rate'], 
                     grad_clip=hyperparams_['grad_clip'], patience=model_['patience'], min_delta=model_['min_delta'])
    model.train_epoch(n_epochs=hyperparams_['epochs'], train_iter=trainprocess.data_iter, val_iter=validprocess.data_iter,
                      save_model_path=model_['save'])

    test_loss = model.evaluate(val_iter=testprocess.data_iter)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

import os
from utils import *
import torch
from model.transformer_mdl import MNTModel
import time
# tensorboard --logdir logs --port 6007 --host 0.0.0.0


def main():
    torch.manual_seed(0)
    config = ReadConfig()
    config.read_config(path="config/config_mdl_sml.yaml", print=True)
    path_cur = os.path.dirname(os.path.abspath(__file__))  # project path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_vocab_dir, trg_vocab_dir = config.src_vocab_dir, config.trg_vocab_dir
    train_processed_dir, valid_processed_dir, test_processed_dir = config.processed_data
    trainsrc_path, valsrc_path, testsrc_path = [os.path.join(
        path_cur, sub_path) for sub_path in config.data_src]
    traintrg_path, valtrg_path, testtrg_path = [os.path.join(
        path_cur, sub_path) for sub_path in config.data_trg]

    if (os.path.isfile(os.path.join(path_cur, src_vocab_dir))) and (os.path.isfile(os.path.join(path_cur, trg_vocab_dir))) and (os.path.isfile(os.path.join(path_cur, train_processed_dir))) and (os.path.isfile(os.path.join(path_cur, valid_processed_dir))) and (os.path.isfile(os.path.join(path_cur, test_processed_dir))):
        lazy_load = True
    else:
        lazy_load = False

    trainprocess = ProcessData(lazy_load)
    validprocess = ProcessData(lazy_load)
    testprocess = ProcessData(lazy_load)

    trainprocess.process(max_len=config.max_len, srcpath=trainsrc_path, trgpath=traintrg_path,
                         vocab_size=config.vocab_size, batch_size=config.batch_size, srcname='EN', trgname='VI',
                         src_vocab_dir=src_vocab_dir, trg_vocab_dir=trg_vocab_dir, train=True, save_dir=train_processed_dir)
    validprocess.process(max_len=config.max_len, srcpath=valsrc_path, trgpath=valtrg_path,
                         vocab_size=config.vocab_size, batch_size=config.batch_size, srcname='EN', trgname='VI',
                         src_vocab_dir=src_vocab_dir, trg_vocab_dir=trg_vocab_dir, train=False, save_dir=valid_processed_dir)
    testprocess.process(max_len=config.max_len, srcpath=testsrc_path, trgpath=testtrg_path,
                        vocab_size=config.vocab_size, batch_size=config.batch_size, srcname='EN', trgname='VI',
                        src_vocab_dir=src_vocab_dir, trg_vocab_dir=trg_vocab_dir, train=False, save_dir=test_processed_dir)

    print("[TRAIN]:%d [TRAIN-ITER]: %d \t[VALID]:%d [VALID-ITER]: %d \t[TEST]:%d [TEST-ITER]: %d"
          % (len(trainprocess.data_prc), len(trainprocess.data_iter), len(validprocess.data_prc),
             len(validprocess.data_iter), len(testprocess.data_prc), len(testprocess.data_iter)))
    st = time.time()    
    model = MNTModel(config, trainprocess.src_vocab_size, trainprocess.trg_vocab_size,
                     trainprocess.vocab_trg, device, config.grad_clip, config.patience, config.min_delta)
    model.train_epoch(n_epochs=config.epochs, train_iter=trainprocess.data_iter, val_iter=validprocess.data_iter,
                      save_model_path=config.path_model_transformer)
    print("Train time: %s"%(time.time()-st))                  
    model.lazyload(config.path_model_transformer, device)
    test_loss = model.evaluate(val_iter=testprocess.data_iter)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

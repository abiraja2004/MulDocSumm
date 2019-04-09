import logging
from setproctitle import setproctitle

import torch

from dataloading import Data
from model import build_VAELSTM
from trainer import Trainer, Trainer_BOW


DATA_DIR = '/home/nlpgpu5/hwijeen/VAE-LSTM/data/'
FILE = 'mscoco'
DEVICE = torch.device('cuda')


setproctitle("(hwijeen) word drop without bow loss")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    data = Data(DATA_DIR, FILE, DEVICE)
    vaeLSTM = build_VAELSTM(len(data.vocab), hidden_dim=600, latent_dim=1100,
                            word_drop=0.25, bow_loss=False, device=DEVICE)
    trainer = Trainer(vaeLSTM, data, lr=0.001, to_record=['recon_loss', 'kl_loss'])
    #trainer = Trainer_BOW(vaeLSTM, data, lr=0.001, to_record=['recon_loss',
    #                                                          'kl_loss', 'bow_loss'])

    trainer.train(num_epoch=10)


import logging
from setproctitle import setproctitle

import torch

from dataloading import MulSumData
from model import build_SelfAttnCVAE
from trainer import Trainer


DATA_DIR = '/home/nlpgpu5/hwijeen/MulDocSumm/data/'
FILE = 'rottentomatoes_prepared'
DEVICE = torch.device('cuda')


setproctitle("(hwijeen) MulDocSumm in progress")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":

    data = MulSumData(DATA_DIR, FILE, 5, DEVICE)
    selfattnCVAE = build_SelfAttnCVAE(len(data.vocab), hidden_dim=600, latent_dim=300,
                               enc_bidirectional=True, device=DEVICE)
    trainer = Trainer(selfattnCVAE, data, lr=0.001, to_record=['recon_loss', 'kl_loss'])
    trainer.train(num_epoch=50)


import logging
from setproctitle import setproctitle

import torch

from dataloading import MulSumData
from model import build_SelfAttnCVAE
from trainer import Trainer
from utils import kl_coef, plot_kl_loss, plot_learning_curve, plot_metrics

setproctitle("(hwijeen) word drop")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":
    DATA_DIR = '/home/nlpgpu5/hwijeen/MulDocSumm/data/'
    FILE = 'rottentomatoes_prepared'
    DEVICE = torch.device('cuda:1')
    EPOCH = 50
    WORD_DROP = 0.2

    data = MulSumData(DATA_DIR, FILE, 5, DEVICE)
    selfattnCVAE = build_SelfAttnCVAE(len(data.vocab), hidden_dim=600, latent_dim=300,
                               enc_bidirectional=True, word_drop=WORD_DROP, device=DEVICE)
    trainer = Trainer(selfattnCVAE, data, lr=0.001, to_record=['recon_loss', 'kl_loss'])
    results = trainer.train(num_epoch=EPOCH, verbose=True)

    plot_learning_curve(results['train_losses'], results['valid_losses'])
    plot_metrics(results['train_metrics'], results['valid_metrics'])
    plot_kl_loss(trainer.stats.stats['kl_loss'])


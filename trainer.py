import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloading import PAD_IDX
from utils import reverse, kl_coef

logger = logging.getLogger(__name__)


class Stats(object):
    def __init__(self, to_record):
        self.to_record = to_record
        self.reset_stats()

    def reset_stats(self):
        self.stats = {name: [] for name in self.to_record}

    def record_stats(self, *args, stat=None):
        stats = self.stats if stat is None else stat
        for name, loss in zip(self.to_record, args):
            stats[name].append(loss.item())

    # do not consider kl coef when reporting average of loss
    def report_stats(self, epoch, step=None, stat=None):
        is_train = stat is None
        stats = self.stats if stat is None else stat
        losses = []
        for name in self.to_record:
            losses.append(np.mean(stats[name]))
        sum_loss = sum(losses)
        if is_train:
            msg = 'loss at epoch {} step {}: {:.2f} ~ recon {:.2f} + kl {:.2f}'\
                .format(epoch, step, sum_loss, losses[0], losses[1])
        else:
            msg = 'valid loss at epoch {}: {:.2f} ~ recon {:.2f} + kl {:.2f}'\
                .format(epoch, sum_loss, losses[0], losses[1])
        logger.info(msg)


class Trainer(object):
    def __init__(self, model, data, lr=0.001, to_record=None):
        self.model = model
        self.data = data
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.stats = Stats(to_record)

    @staticmethod
    def kl_div_two_normal(p_params, q_params):
        mu1, log_var1 = p_params
        mu2, log_var2 = q_params
        return torch.sum(((log_var1 - log_var2) + 1 \
                           - (log_var1.exp() / log_var2.exp()) \
                           - ((mu1 - mu2).pow(2) / log_var2.exp())) * -0.5, dim=1).mean()

    def compute_loss(self, batch, total_step=0):
        variational_params, prior_params, recon_logits = self.model(batch)
        B, L, _ = recon_logits.size()
        target, _ = batch.summ
        recon_loss = self.criterion(recon_logits.view(B*L, -1), target.view(-1))
        kl_loss = 0
        for var_par, prior_par in zip(variational_params.values(), prior_params.values()):
            kl_loss += self.kl_div_two_normal(var_par, prior_par) / len(prior_params)
        coef = kl_coef(total_step) # kl annlealing
        return recon_loss, kl_loss, coef

    def train(self, num_epoch, closed_test=False):
        total_step = 0 # for KL annealing
        for epoch in range(1, num_epoch, 1):
            # TODO: print stats on what basis?
            #self.stats.reset_stats()
            for step, batch in enumerate(self.data.train_iter, 1):
                total_step += 1
                recon_loss, kl_loss, coef = self.compute_loss(batch, total_step)
                loss = recon_loss + coef * kl_loss
                self.stats.record_stats(recon_loss, kl_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                if step % 10 == 0:
                    self.stats.report_stats(epoch, step=step)

            if closed_test:
                with torch.no_grad():
                    self.inference(data_type='train')
            # evaluate on dev set at the end of every epoch
            with torch.no_grad():
                valid_stats = {name: [] for name in self.stats.to_record}
                for batch in self.data.valid_iter:
                    recon_loss, kl_loss, _= self.compute_loss(batch)
                    self.stats.record_stats(recon_loss, kl_loss, stat=valid_stats)
                self.stats.report_stats(epoch, stat=valid_stats)
                self.inference(data_type='valid')

    def inference(self, data_type):
        if data_type == 'train':
            data_iter = self.data.train_iter
        elif data_type == 'valid':
            data_iter = self.data.valid_iter
        elif data_type == 'test':
            data_iter = self.data.test_iter

        random_idx = random.randint(0, len(data_iter))
        for idx, batch in enumerate(data_iter): # to get a random batch
            if idx == random_idx: break
        paraphrased = self.model.inference(batch.orig)
        paraphrased = reverse(paraphrased, self.data.vocab)
        original = reverse(batch.orig[0], self.data.vocab)
        reference = reverse(batch.para[0], self.data.vocab) \
            if data_type == 'valid' else None
        print('sample paraphrases in {} data'.format(data_type))
        if data_type == 'valid':
            for orig, para, ref in zip(original, paraphrased, reference):
                print(orig, '\t => \t', para)
                print('\t\t\t reference: ', ref)
        else:
            for orig, para in zip(original, paraphrased):
                print(orig, '\t => \t', para)

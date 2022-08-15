#!/usr/bin/python


"""Energy-based training of a flow model on an atomistic system."""

from typing import Callable, Dict, Tuple
from absl import app
from absl import flags
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from nflows.flows import Flow
from experiments.configs import get_config
from experiments.utils import *

flags.DEFINE_enum(name='system', default='_16',
                  enum_values=['_4', '_8', '_16', '_32', '_64'
                               ], help='System and number of atoms to train.')
# flags.DEFINE_integer(name='val_freq', default=50)
flags.DEFINE_integer(name='max_iter', default=int(10**6), help='Max iteration of training.')
# flags.DEFINE_integer(name='seed', default=2020, help="Seed of random generation.")
flags.DEFINE_bool(name='reduce_on_plateau', default=True, help='Reduce on plateau or multi-step lr')
flags.DEFINE_bool(name='resume', default=False, help='Resume from previous model.')
flags.DEFINE_string(name='tag', default='', help='Tag of the saved model.')
flags.DEFINE_string(name='log_root', default='./logs', help='Log dir.')

FLAGS = flags.FLAGS


def _num_particles(system: str) -> int:
    return int(system.split('_')[-1])


def _get_loss(
        model: Flow,
        trans: Module,
        energy_fn: Callable,
        beta: Tensor,
        num_samples: int) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Returns the loss and stats."""
    samples, log_prob = model.sample_and_log_prob(
        num_samples=num_samples)
    samples_3D = trans(samples)
    energies = energy_fn(samples_3D, model._distribution.mol)
    energy_loss = torch.mean(beta * energies + log_prob)

    loss = energy_loss
    stats = {
        'energy': energies,
        'model_log_prob': log_prob,
        'target_log_prob': -beta * energies
    }
    return loss, stats


def main(_):
    system = FLAGS.system
    config = get_config(_num_particles(system))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = config.state
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(root=FLAGS.log_root,
                              prefix='IDP_NF', tag=FLAGS.tag)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    # log_hyperparams(writer, config)

    # Model
    logger.info('Building model...')
    model, trans = config.model['constructor'](
        lower=state.lower,
        upper=state.upper,
        **config.model['kwargs'])
    if FLAGS.resume:
        ckpt_resume = CheckpointManager(
            './pretrained', logger=logger).load_latest()
        logger.info(f'Resuming from iteration {ckpt_resume["iteration"]}')
        model.load_state_dict(ckpt_resume['state_dict'])

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(),
                     lr=config.train['learning_rate'])
    if FLAGS.reduce_on_plateau:
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=config.train['learning_rate_decay_factor'])
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.train['learning_rate_decay_steps'],
            gamma=config.train['learning_rate_decay_factor'])

    def train(iter: int):
        model.train()
        optimizer.zero_grad()
        loss, stats = _get_loss(
            model=model,
            trans=trans,
            energy_fn=config.energy,
            beta=state.beta,
            num_samples=config.train.batch_size,
        )
        print(loss.device)
        loss.backward()
        optimizer.step()
        metrics = {
            'loss': loss,
            'energy': torch.mean(stats['energy']),
            'model_entropy': -torch.mean(stats['model_log_prob']),
        }
        logger.info('[Train] Iter %04d | Loss %.6f | Energy %d | Entropy %d ' % (
            iter, loss.item(), metrics['energy'], metrics['model_entropy']))
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter)
        writer.add_scalar('train/energy', metrics['energy'], iter)
        writer.add_scalar('train/model_entropy',
                          metrics['model_entropy'], iter)
        writer.flush()

    def validate(iter: int, is_save=False):
        with torch.no_grad():
            model.eval()
            loss, stats = _get_loss(
                model=model,
                trans=trans,
                energy_fn=config.energy,
                beta=state.beta,
                num_samples=config.test.batch_size,
            )
            scheduler.step(loss)
            metrics = {
                'loss': loss,
                'energy': torch.mean(stats['energy']),
                'model_entropy': -torch.mean(stats['model_log_prob']),
            }
            logger.info('[Train] Iter %04d | Loss %.6f | Energy %d | Entropy %d ' % (
                iter, loss.item(), metrics['energy'], metrics['model_entropy']))
            writer.add_scalar('val/loss', loss, iter)
            writer.add_scalar('train/energy', metrics['energy'], iter)
            writer.add_scalar('train/model_entropy',
                              metrics['model_entropy'], iter)
            writer.flush()
            if is_save:
                ckpt_mgr.save(model, FLAGS, loss, iter)

    step = 0
    if FLAGS.resume:
        step = ckpt_resume["iteration"] + 1
    logger.info(f'Start training from iter {step}.')
    while step < FLAGS.max_iter:
        # Training update.
        train(step)

        if (step % config.test.test_every) == 0:
            validate(step, is_save=True)

        step += 1

    logger.info('Done')


if __name__ == '__main__':
    app.run(main)

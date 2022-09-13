#!/usr/bin/python


"""Energy-based training of a flow model on an atomistic system."""

from logging import Logger
from typing import Callable, Dict, Tuple
from absl import app
from absl import flags
from pathlib import Path
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from nflows.flows import Flow
from experiments.configs import get_config
from experiments.utils import *
from models.pdb_files import save_pdb

# flags.DEFINE_enum(name='system', default='A_16_2020',
#                   enum_values=['L_4_2020', 'A_8_2020', 'A_14_2019', 'A_16_2020', 'A_16_42'
#                                ], help='System and number of atoms to train.')
flags.DEFINE_string(name='system', default='A_16_42', help='System and number of atoms to train.')
flags.DEFINE_integer(name='max_iter', default=int(2000), help='Max iteration of training.')
flags.DEFINE_bool(name='reduce_on_plateau', default=True, help='Reduce on plateau or multi-step lr')
flags.DEFINE_bool(name='resume', default=False, help='Resume from previous model.')
flags.DEFINE_string(name='tag', default='', help='Tag of the saved model.')
flags.DEFINE_string(name='log_root', default='./logs', help='Log dir.')
flags.DEFINE_integer(name='start_mol', default=0, help='Start id of molecules to train.')

FLAGS = flags.FLAGS


def _split_system(system: str) -> Dict:
    sys, num_atoms, seed = system.split('_')
    if sys == 'L':
        molecular = 'lignin'
    elif sys == 'A':
        molecular = 'alkane'
    elif sys == 'C':
        molecular = 'chignolin'
    elif sys == 'P':
        molecular = 'pdb'
    return {
        'num_atoms': int(num_atoms), 
        'molecular': molecular, 
        'seed': int(seed)}


def _get_loss(
        model: Flow,
        trans: Module,
        energy_fn: Callable,
        beta: Tensor,
        num_samples: int,
        logger: Logger,
        mode='train') -> Tuple[Tensor, Dict[str, Tensor]]:
    """Returns the loss and stats."""
    samples, log_prob = model.sample_and_log_prob(
        num_samples=num_samples)
    samples_3D = trans(samples)
    energies = energy_fn(samples_3D, model._distribution.mol, mode)
    # logger.debug(f'Energy after model: {energies}')
    energy_loss = torch.mean(beta * energies + log_prob)

    loss = energy_loss
    stats = {
        'energy': energies,
        'model_log_prob': log_prob,
        'target_log_prob': -beta * energies
    }
    return loss, stats


def work(system='None'):
    if system == 'None':
        system = FLAGS.system

    # Logging
    log_dir = get_new_log_dir(root=FLAGS.log_root,
                              prefix=system, tag=FLAGS.tag)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(
        './pretrained',
        logger=logger,
        log_dir=log_dir)
    # log_hyperparams(writer, config)

    # Config
    config = get_config(logger, **_split_system(system))
    state = config.state
    seed_all(config.train.seed)

    # Model
    logger.info(f'System {system} start working!')
    logger.info('Building model...')
    model, trans = config.model['constructor'](
        lower=state.lower,
        upper=state.upper,
        **config.model['kwargs'])
    logger.info(f'Torsion angles: {model._distribution.torsion_angles}')
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
            factor=config.train['learning_rate_decay_factor'],
            patience=config.train['patience'],
            verbose=True)
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones=config.train['learning_rate_decay_steps'],
            gamma=config.train['learning_rate_decay_factor'],
            verbose=True)

    def train(iter: int):
        model.train()
        loss, stats = _get_loss(
            model=model,
            trans=trans,
            energy_fn=config.energy,
            beta=state.beta,
            num_samples=config.train.batch_size,
            logger=logger,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        trans.zero_grad()
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
    
    def validate(iter: int, best_loss=10000):
        with torch.no_grad():
            model.eval()
            loss, stats = _get_loss(
                model=model,
                trans=trans,
                energy_fn=config.energy,
                beta=state.beta,
                num_samples=config.test.batch_size,
                logger=logger,
                mode='test'
            )
            scheduler.step(loss)
            metrics = {
                'loss': loss,
                'energy': torch.mean(stats['energy']),
                'model_entropy': -torch.mean(stats['model_log_prob']),
            }
            logger.info('[ Val ] Iter %04d | Loss %.6f | Energy %d | Entropy %d ' % (
                iter, loss.item(), metrics['energy'], metrics['model_entropy']))
            writer.add_scalar('val/loss', loss, iter)
            writer.add_scalar('val/energy', metrics['energy'], iter)
            writer.add_scalar('val/model_entropy',
                              metrics['model_entropy'], iter)
            writer.flush()
            if config.test['save_mode'] == 'threshold' and loss < config.test['save_threshold']:
                ckpt_mgr.save(model, loss, iter)
            elif config.test['save_mode'] == 'best' and loss < best_loss:
                best_loss = loss
                save_pdb(model._distribution.mol, config.test['file_name'])
            return best_loss

    step = 0
    best_loss = torch.tensor(10000)
    if FLAGS.resume:
        step = ckpt_resume["iteration"] + 1
    logger.info(f'Start training from iter {step}.')
    while step < FLAGS.max_iter:
        # Training update.
        train(step)

        if (step % config.test.test_every) == 0:
            best_loss = validate(step, best_loss)

        step += 1

    logger.info(f'System {system} done!')

def main(_):
    pathlist = Path('cleaned_geom_mols').glob('mol_*.pdb')[FLAGS.start_mol:]
    for path in pathlist:
        mol_id = int(str(path).split('/')[-1].split('_')[-1].split('.')[0])
        work(f'P_{mol_id}_2022')

if __name__ == '__main__':
    app.run(main)

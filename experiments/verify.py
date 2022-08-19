from ml_collections.config_dict import ConfigDict
from experiments.configs import get_config
from experiments.utils import *
from matplotlib import pyplot as plt
import numpy as np

log_dir = get_new_log_dir(root='./logs',
                          prefix='', tag='verify')
logger = get_logger('verify', log_dir)
config_8_2020 = get_config(8, logger)
config_16_2020 = get_config(16, logger)
config_14_2019 = get_config(14, logger)
config_14_2019.train['seed'] = 2019

def verify_model(config: ConfigDict, logdir: str):

    # Config
    state = config.state
    seed_all(config.train.seed)

    fig, axs = plt.subplots(2, 1,
                        figsize =(10, 7),
                        tight_layout = True)
    bins = np.arange(start=-5, stop=75, step=2)
    # Model
    logger.info(f'Building model for {logdir}')
    model, trans = config.model['constructor'](
        lower=state.lower,
        upper=state.upper,
        **config.model['kwargs'])
    logger.info(f'Before training, {logdir}')
    with torch.no_grad():
        model.eval()
        samples, log_prob = model.sample_and_log_prob(
            num_samples=config.test['batch_size'])
        samples_3D = trans(samples)
        energies = config.energy(samples_3D, model._distribution.mol, 'verify')
        logger.info(energies.sort().values)
        axs[0].hist(energies.sort().values.cpu().detach().numpy(), bins = bins, histtype='bar', ec='black')
        axs[0].set_title('Before train')


    logger.info(f'After training, {logdir}')
    ckpt_resume = CheckpointManager(
        './pretrained', logger=logger, log_dir=logdir).load_latest()
    model.load_state_dict(ckpt_resume['state_dict'])
    with torch.no_grad():
        model.eval()
        samples, log_prob = model.sample_and_log_prob(
            num_samples=config.test['batch_size'])
        samples_3D = trans(samples)
        energies = config.energy(samples_3D, model._distribution.mol, 'verify')
        logger.info(energies.sort().values)
        axs[1].hist(energies.sort().values.cpu().detach().numpy(), bins = bins, histtype='bar', ec='black')
        axs[1].set_title('After train')
    # axs.set_title(f'{logdir}')
    # plt.rcParams['savefig.dpi'] = 300
    # plt.rcParams['figure.dpi'] = 300
    plt.savefig(f"figs/{logdir}.png")

verify_model(config_8_2020, '2022_08_18__11_26_54_A8_2020')
verify_model(config_14_2019, '2022_08_18__08_09_51_A14')
verify_model(config_16_2020, '2022_08_18__11_02_26_A16_2020')

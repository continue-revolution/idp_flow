import os
from ml_collections.config_dict import ConfigDict
from experiments.configs import get_config
from experiments.utils import *
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
from rdkit import Chem
from rdkit.Chem import TorsionFingerprints

log_dir = get_new_log_dir(root='./logs',
                          prefix='', tag='verify')
logger = get_logger('verify', log_dir)
config_8_2020 = get_config(8, logger)
config_16_2020 = get_config(16, logger)
config_14_2019 = get_config(14, logger)
config_16_42 = get_config(16, logger)
config_14_2019.train['seed'] = 2019
config_16_42.train['seed'] = 42


def tfd_matrix(mol: Chem.Mol) -> np.array:
    """Calculates the TFD matrix for all conformers in a molecule.
    """
    tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False)
    n = int(np.sqrt(len(tfd)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n, n))
    matrix[idx] = tfd
    matrix += np.transpose(matrix)
    return matrix

def verify_property(mol: Chem.Mol, logdir: str):
    tfd = tfd_matrix(mol)
    sns.heatmap(tfd)
    plt.savefig(f"figs/{logdir}/tfd_heatmap.png")
    plt.clf()

    nonring, _ = TorsionFingerprints.CalculateTorsionLists(mol)
    nonring = [list(atoms[0]) for atoms, ang in nonring]

    torsion_angles = []
    for conf_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(conf_id)
        conf_torsions = [Chem.rdMolTransforms.GetDihedralDeg(conf, *tors) for tors in nonring]
        torsion_angles.append(conf_torsions)
    torsion_angles = np.array(torsion_angles)

    corrs = np.corrcoef(torsion_angles.T) # want correlation between angles *not* conformers
    corrs[np.abs(corrs) < 0.01] = 0
    corrs = np.abs(corrs)
    sns.heatmap(corrs, cmap="mako")
    plt.savefig(f"figs/{logdir}/corrs_heatmap.png")
    plt.clf()

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
        './pretrained', logger=logger, log_dir=logdir).load_best()
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
    os.makedirs('figs', exist_ok=True)
    os.makedirs(f'figs/{logdir}', exist_ok=True)
    plt.savefig(f"figs/{logdir}/energy.png")
    plt.clf()
    verify_property(model._distribution.mol, logdir)

# verify_model(config_8_2020, '2022_08_18__11_26_54_A8_2020')
# verify_model(config_14_2019, '2022_08_18__08_09_51_A14')
# verify_model(config_16_2020, '2022_08_18__11_02_26_A16_2020')
verify_model(config_16_2020, '2022_08_23__11_38_49_A16_2020')
verify_model(config_16_42, '2022_08_23__11_26_58_A16_42')

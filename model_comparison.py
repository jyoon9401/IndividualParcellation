"""
DCBC evaluation of individual parcellations obtained from 
HBP, Dual Regression and Dictionary-Learning models

authors: Ana Luisa Pinho, Jennifer Yoon
emails: agrilopi@uwo.ca, jyoon94@uwo.ca

Compatibility: Python 3.9.20 / PyTorch 2.5.1+cu118
"""

from pathlib import Path

import torch as pt
import numpy as np
import pandas as pd
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as hev
import FusionModel.evaluate as fev
import IndividualParcellation.scripts.dual_regression as dr
import IndividualParcellation.scripts.utils_dictionary as dl

from sklearn.model_selection import KFold
from DCBC.utilities import compute_dist
from global_config import BASE_DIR, DEVICE
from utils import prob_arrmodel, cv_cosine_ols


# ###################### FUNCTIONS #####################################

def get_alternative_iparcels(model, tr_data, dl_method='online', 
                             dl_init='kmeans', Ug=None, V=None, 
                             n_parcels=20, alpha=.01, prior_type='prob'):
        
    if model == 'DR':
        parcels = dr.get_iparcel_dualreg(tr_data, Ug=Ug, V=V, 
                                         prior_type=prior_type)
    else:
        assert model == 'DL'
        parcels, _, _ = dl.get_iparcel_dictlearning(
            tr_data, n_parcels=n_parcels, vinit_type=dl_init, dict_init=V, 
            method=dl_method, alpha=alpha, l1_ratio=.5, write_dir='/tmp')

    return parcels


def evaluate_parcels(indiv_parcels, dtest, atlas):

    # distance matrix
    dist = compute_dist(atlas.world.T, resolution=1)

    if type(indiv_parcels) is np.ndarray:
        indiv_parcels = pt.tensor(indiv_parcels)

    if type(dtest) is np.ndarray:
        dtest = pt.tensor(dtest)

    dim = 1 if indiv_parcels.dim() == 3 else 0
    pindiv = pt.argmax(indiv_parcels, dim=dim) + 1
    dcbc = fev.calc_test_dcbc(pindiv, dtest, dist)

    return dcbc


# ####################### INPUTS #######################################
    
## Atlas
atlas_name = 'fs32k'  # 'fs32k' # 'MNISymC3'
# atlas_fname = 'asym_Md_space-MNISymC3_K-17_probseg.nii.gz'
atlas_fname = 'DU15NET_GroupProb_fsLR_32k.dscalar.nii'
sym_type = 'asym' # 'sym'

# Dataset parameters
dataset = 'MDTB'

# Dictionary-Learning hyperparameters
alphas = np.logspace(-4, 0, 9)

# Evaluation metrics
eval_metric = 'dcbc'
# eval_metric = 'cosine_average'
# eval_metric = 'cosine_expected'

# Paths
main_dir = Path.cwd()
eval_dir = Path(str(Path(main_dir, 'evaluation')))
atlas_path = Path(str(Path('atlases', atlas_fname)))

# Number of clusters
n_clusters = 15 # 15, 17, 32

# ######################## RUN #########################################

if __name__ == "__main__":
    # Load Arrangement Model
    atlas, _, U, ar_model = prob_arrmodel(atlas_name, str(atlas_path), 
                                          sym_type)
    
    # Get the condition effects for every run
    data1r, info1r, _ = ds.get_dataset(BASE_DIR, dataset, atlas=atlas.name, 
                                       subj=None, sess='ses-s1', 
                                       type='CondRun')
    data2r, info2r, _ = ds.get_dataset(BASE_DIR, dataset, atlas=atlas.name,
                                       subj=None, sess='ses-s2', 
                                       type='CondRun')
    
    # Get the condition effects for every half session
    data1h, info1h, _ = ds.get_dataset(BASE_DIR, dataset, atlas=atlas.name,
                                       subj=None, sess='ses-s1', 
                                       type='CondHalf')
    data2h, info2h, _ = ds.get_dataset(BASE_DIR, dataset, atlas=atlas.name,
                                       subj=None, sess='ses-s2', 
                                       type='CondHalf')
    
    # Split data for train and test in runs
    tdata1r, _, _, sub_ind = fm.prep_datasets(
        data1r, info1r.run,
        info1r['cond_num_uni'].values, info1r['run'].values,
        join_sess=False, join_sess_part=False)
    tdata2r, _, _, _ = fm.prep_datasets(
        data2r, info2r.run,
        info2r['cond_num_uni'].values, info2r['run'].values,
        join_sess=False, join_sess_part=False)
    
    # Split data for train and test in half sessions
    tdata1h, _, _, _ = fm.prep_datasets(
        data1h, info1h.half,
        info1h['cond_num_uni'].values, info1h['half'].values,
        join_sess=False, join_sess_part=False)
    tdata2h, _, _, _ = fm.prep_datasets(
        data2h, info2h.half,
        info2h['cond_num_uni'].values, info2h['half'].values,
        join_sess=False, join_sess_part=False)
    
    # Append train and test data
    tdata1 = [np.array(tdata1r)] + [np.array(tdata2h)]
    tdata2 = [np.array(tdata2r)] + [np.array(tdata1h)]
    tdata_set = [tdata1] + [tdata2]

    info1 = [info1r] + [info2h]
    info2 = [info2r] + [info1h]
    info = [info1] + [info2]

    evals_hbp_group = []
    evals_hbp_full = []
    evals_hbp_data = []
    evals_dr_ugica = []
    evals_dr_uhbp = []
    evals_dr_vhbp = []
    evals_dl_online = []
    evals_dl_sparse = []
    train_sessions_id = []
    test_sessions_id = []
    train_runs_id = []
    subjects = []

    for t, tdata in enumerate(tdata_set):
        train_session = tdata[0]
        info_train_session = info[t][0]        

        for _, train_runs_idx in KFold(
            n_splits=8, shuffle=False).split(train_session):

            train_data = train_session[train_runs_idx]
            info_train_data = info_train_session[
                info_train_session.run.isin(train_runs_idx + 1)]
            test_data = tdata[1]
            info_test = info[t][1]

            cond_vtrain1 = np.tile(np.arange(1, train_data.shape[2] + 1), 
                                   train_data.shape[0])
            part_vtrain1 = np.ravel([
                np.repeat(tr_run + 1, train_data.shape[2]) 
                for tr_run in train_runs_idx])

            train_data = np.swapaxes(train_data, 0, 1)
            train_data_concat = np.mean(train_data, axis=1)
            train_data = np.reshape(train_data, (
                train_data.shape[0], 
                train_data.shape[1] * train_data.shape[2], 
                train_data.shape[3]))
            
            test_data = np.swapaxes(test_data, 0, 1)
            test_data_concat = np.mean(test_data, axis=1)
            test_data = np.reshape(test_data, (
                test_data.shape[0], 
                test_data.shape[1] * test_data.shape[2], 
                test_data.shape[3]))

            # Mean centering across conditions
            train_mean = train_data.mean(axis=1, keepdims=True)
            train_mean_concat = train_data_concat.mean(axis=1, keepdims=True)
            test_mean = test_data.mean(axis=1, keepdims=True)
            test_mean_concat = test_data_concat.mean(axis=1, keepdims=True)

            mean_centered_train = train_data - train_mean
            mean_centered_train_concat = train_data_concat - train_mean_concat
            mean_centered_test = test_data - test_mean
            mean_centered_test_concat = test_data_concat - test_mean_concat

            cond_vtrain = info_train_data['cond_num_uni'].values
            part_vtrain = info_train_data['run'].values
            cond_vtest = info_test['cond_num_uni'].values
            part_vtest = info_test['half'].values
            sub_index = sub_ind[0]

            # ******** GET HBP INDIVIDUAL PARCELLATIONS ****************
            Ui_full, _, M_train = fm.get_indiv_parcellation(
                ar_model, atlas, [mean_centered_train], [cond_vtrain], 
                [part_vtrain], [sub_index])
            V_em = M_train.emissions[0].V

            # Get data-only individual parcellations
            emloglik = M_train.emissions[0].Estep()
            Ui_data = pt.softmax(emloglik, dim=1)
            # Alternative to extract prior + individual parcellations
            # Ui_full, _ = M_train.arrange.Estep(emloglik)

            # # # *** GET DUAL-REGRESSION INDIVIDUAL PARCELLATIONS *********
            U_ica = dr.group_ica(
                mean_centered_train_concat, n_components=n_clusters)
            parcels_dr_ugica = get_alternative_iparcels(
                'DR', mean_centered_train_concat, Ug=U_ica, 
                n_parcels=n_clusters)
            parcels_dr_uhbp = get_alternative_iparcels(
                'DR', mean_centered_train_concat, Ug=U.T, 
                n_parcels=n_clusters)
            parcels_dr_vhbp = get_alternative_iparcels(
                'DR', mean_centered_train_concat, V=V_em, 
                n_parcels=n_clusters)
           
            # # # ** GET DICTIONARY-LEARNING INDIVIDUAL PARCELLATIONS ******
            alpha_parcel_online = []
            alpha_parcel_sparse = []
            for alpha in alphas:
                parcels_dl_online = get_alternative_iparcels(
                    'DL', mean_centered_train_concat, dl_method='online', 
                    dl_init='hbp', V=V_em, n_parcels=n_clusters, alpha=alpha)
                parcels_dl_sparse = get_alternative_iparcels(
                    'DL', mean_centered_train_concat, dl_method='sparse', 
                    dl_init='hbp', V=V_em, n_parcels=n_clusters, alpha=alpha)
                alpha_parcel_online.append(parcels_dl_online)
                alpha_parcel_sparse.append(parcels_dl_sparse)

            if eval_metric == 'dcbc':
                # ****** EVALUATE HBP PARCELLATIONS ON THE TEST SET ********
                eval_hbp_group = evaluate_parcels(U, mean_centered_test, 
                                                  atlas)
                eval_hbp_full = evaluate_parcels(Ui_full, mean_centered_test, 
                                                 atlas)
                eval_hbp_data = evaluate_parcels(Ui_data, mean_centered_test, 
                                                 atlas)

                # * EVALUATE DUAL-REGRESSION PARCELLATIONS ON THE TEST SET *
                eval_dr_ugica = evaluate_parcels(parcels_dr_ugica, 
                                                 mean_centered_test, atlas)
                eval_dr_uhbp = evaluate_parcels(parcels_dr_uhbp, 
                                                mean_centered_test, atlas)
                eval_dr_vhbp = evaluate_parcels(parcels_dr_vhbp, 
                                                mean_centered_test, atlas)

                # ****** EVALUATE DICTIONARY-LEARNING PARCELLATIONS ********
                # ***************** ON THE TEST SET ************************
                alpha_eval_online = []
                alpha_eval_sparse = []
                for a in np.arange(len(alphas)):
                    eval_dl_online = evaluate_parcels(
                        alpha_parcel_online[a], mean_centered_test, atlas)            
                    eval_dl_sparse = evaluate_parcels(
                        alpha_parcel_sparse[a], mean_centered_test, atlas)
                    alpha_eval_online.append(eval_dl_online.tolist())
                    alpha_eval_sparse.append(eval_dl_sparse.tolist())

            else:
                assert eval_metric[:6] == 'cosine'

                # Type casting of test data
                mean_centered_test = mean_centered_test.astype(np.float32)
                mean_centered_test_concat = \
                    mean_centered_test_concat.astype(np.float32)
                             
                # ****** EVALUATE HBP PARCELLATIONS ON THE TEST SET ********

                # Get HBP full model of test data
                _, _, M_test = fm.get_indiv_parcellation(
                    ar_model, atlas, [mean_centered_test], [cond_vtest], 
                    [part_vtest], [sub_index])
                
                # Type casting and ensure compatibility between GPU 
                # and CPU computations
                U_group = pt.from_numpy(U).to(pt.float32).to(DEVICE)
                mean_centered_test_tensor = pt.from_numpy(
                    mean_centered_test).to(dtype=pt.float32).to(DEVICE)

                # Evaluate
                eval_hbp_group = hev.calc_test_error(
                    M_test, mean_centered_test_tensor, [U_group], 
                    coserr_type=eval_metric[7:])[0]
                eval_hbp_full = hev.calc_test_error(
                    M_test, mean_centered_test_tensor, [Ui_full], 
                    coserr_type=eval_metric[7:])[0]
                eval_hbp_data = hev.calc_test_error(
                    M_test, mean_centered_test_tensor, [Ui_data], 
                    coserr_type=eval_metric[7:])[0]

                # * EVALUATE DUAL-REGRESSION PARCELLATIONS ON THE TEST SET *
                eval_dr_ugica = cv_cosine_ols(
                    mean_centered_test_concat, parcels_dr_ugica, 
                    coserr_type=eval_metric[7:])
                eval_dr_uhbp = cv_cosine_ols(
                    mean_centered_test_concat, parcels_dr_uhbp, 
                    coserr_type=eval_metric[7:])
                eval_dr_vhbp = cv_cosine_ols(
                    mean_centered_test_concat, parcels_dr_vhbp, 
                    coserr_type=eval_metric[7:])

                # ****** EVALUATE DICTIONARY-LEARNING PARCELLATIONS ********
                # ***************** ON THE TEST SET ************************
                alpha_eval_online = []
                alpha_eval_sparse = []
                for a in np.arange(len(alphas)):
                    eval_dl_online = cv_cosine_ols(
                        mean_centered_test_concat, alpha_parcel_online[a], 
                        coserr_type=eval_metric[7:])
                    eval_dl_sparse = cv_cosine_ols(
                        mean_centered_test_concat, alpha_parcel_sparse[a], 
                        coserr_type=eval_metric[7:])
                    alpha_eval_online.append(eval_dl_online.tolist())
                    alpha_eval_sparse.append(eval_dl_sparse.tolist())

            evals_hbp_group.extend(eval_hbp_group)
            evals_hbp_full.extend(eval_hbp_full)
            evals_hbp_data.extend(eval_hbp_data)

            evals_dr_ugica.extend(eval_dr_ugica)
            evals_dr_uhbp.extend(eval_dr_uhbp)
            evals_dr_vhbp.extend(eval_dr_vhbp)

            if not evals_dl_online:
                evals_dl_online = alpha_eval_online
                evals_dl_sparse = alpha_eval_sparse
            else:
                evals_dl_online = np.append(
                    evals_dl_online, alpha_eval_online, axis=1).tolist()
                evals_dl_sparse = np.append(
                    evals_dl_sparse, alpha_eval_sparse, axis=1).tolist()

            if t == 0:
                train_session_id = 1
                test_session_id = 2
            else:
                assert t == 1
                train_session_id = 2
                test_session_id = 1

            train_runs_idses = train_runs_idx + 1

            train_sessions_id.extend(np.repeat(train_session_id, 
                                               len(sub_index)).tolist())
            test_sessions_id.extend(np.repeat(test_session_id, 
                                              len(sub_index)).tolist())
            train_runs_id.extend([train_runs_idses.tolist() 
                                  if len(train_runs_idses) > 1 
                                  else train_runs_idses.tolist()[0] 
                                  for i in np.arange(len(sub_index))])

            subjects.extend(np.array(sub_index + 1).tolist())

            assert len(evals_hbp_group) == len(evals_hbp_full) == \
                len(evals_hbp_data) == len(evals_dr_ugica) == \
                len(evals_dr_uhbp) == len(evals_dr_vhbp) == \
                len(evals_dl_online[0]) == len(evals_dl_sparse[0]) == \
                len(train_sessions_id) == len(test_sessions_id) == \
                len(train_runs_id)

    # *********************** SAVE ***********************************
    table = np.vstack((
        subjects, 
        np.array([tensor.item() for tensor in evals_hbp_group]),
        np.array([tensor.item() for tensor in evals_hbp_full]),
        np.array([tensor.item() for tensor in evals_hbp_data]),
        np.array([tensor.item() for tensor in evals_dr_ugica]),
        np.array([tensor.item() for tensor in evals_dr_uhbp]),
        np.array([tensor.item() for tensor in evals_dr_vhbp]),
        evals_dl_online,
        evals_dl_sparse
    )).T
    
    df = pd.DataFrame(table, columns=[
        'Subject', 
        'HBP Group', 'HBP Group + Individual', 'HBP Individual', 
        'DR GICA-U', 'DR HBP-U', 'NNLS', 
        'DL Online alpha=1e-4', 'DL Online alpha=3.16e-4', 
        'DL Online alpha=1e-3', 'DL Online alpha=3.16e-3',
        'DL Online alpha=1e-2', 'DL Online alpha=3.16e-2',
        'DL Online alpha=1e-1', 'DL Online alpha=3.16e-1',
        'DL Online alpha=1e0', 
        'Sparse alpha=1e-4', 'Sparse alpha=3.16e-4', 
        'Sparse alpha=1e-3', 'Sparse alpha=3.16e-3',
        'Sparse alpha=1e-2', 'Sparse alpha=3.16e-2',
        'Sparse alpha=1e-1', 'Sparse alpha=3.16e-1',
        'Sparse alpha=1e0'
        ])
    
    df.insert(1, 'Train Session', train_sessions_id)
    df.insert(2, 'Test Session', test_sessions_id)
    df.insert(3, 'Train Runs', train_runs_id)
    
    df['Subject'] = df['Subject'].astype(int)
    
    # Create output folder, if it does not exist
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(str(Path(
        eval_dir, eval_metric.replace('_', '-') + '_evaluation_' + dataset + 
        '_cvrun.tsv')), sep='\t', index=False)
import sys
sys.path.append('/home/ec2-user/ember')
sys.path.append('/home/ec2-user/pyDNMFk')
import ember
import os
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import category_encoders as ce
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import Counter, namedtuple, OrderedDict
from scipy.io import savemat, loadmat
#from pyDNMFk.runner import pyDNMFk_Runner
import subprocess
import pickle
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import inspect
from scipy.optimize import nnls


def parse_args():

    parser = ArgumentParser()

    @dataclass
    class Options:
        """ Help string for this group of command-line arguments """
        train: bool = False     # Discover latent features through pyDNMFk and build signature archive
        inference: bool = False # Run inference by NNLS projection to signature archive
        verbose: bool = False   # print run-time information

    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()

    return args, parser  # return parser to access help message


def load_data(data_set_type='train', data_dir='/home/ec2-user/data/ember2018/', ndim=645):  # ndim is ember feature dimension
    metadata_dataframe = ember.read_metadata('/home/ec2-user/data/ember2018/')
    df = metadata_dataframe[(metadata_dataframe['subset'] == data_set_type) & (metadata_dataframe['label'] == 1)]
    df = df.dropna()
    
    df_tmp = df['avclass'].value_counts().to_frame()
    most_common_families = sorted(list(df_tmp.head(10).index))
    
    df_most_common = df[df['avclass'].isin(most_common_families)][['avclass']]
    
    filename = 'X_' + data_set_type + '.dat'
    datapath = os.path.join(data_dir, filename)
    
    most_common_index = df_most_common.index
    if data_set_type == 'train':
        nrof_samples = 800_000
    else:
        nrof_samples = 200_000
        most_common_index -= 800_000
        df_most_common.set_index(most_common_index, inplace=True)
    
    X = np.memmap(datapath, dtype=np.float32, mode="r", shape=(nrof_samples, ndim))
    
    X_select = X[most_common_index,:]
    
    df_X = pd.DataFrame(X_select)
    df_X = df_X.add_prefix('feat_')
    df_X.set_index(most_common_index, inplace=True)
    df_X = pd.concat([df_X, df_most_common], axis=1)
    
    df_sampled = df_X.groupby('avclass').apply(lambda x: x.sample(n=1_000, random_state=42))
    
    df_shuffled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    X_shuffled = df_shuffled.drop(columns='avclass').to_numpy()
    
    return df_shuffled, X_shuffled


def scale_data(data, mean_values=None, std_values=None, scaler=None, verbose=True):
    scaled_data = data.copy()
    
    if mean_values is None:
        mean_values = np.mean(data, axis=0)
    
    if std_values is None:
        std_values  = np.std(data, axis=0)
    
    for row in tqdm(range(scaled_data.shape[0])):
        for col in range(scaled_data.shape[1]):
            if np.abs(scaled_data[row, col] - mean_values[col]) > 3*std_values[col]:
                scaled_data[row, col] = 3*std_values[col]
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(scaled_data)
    
    scaled_data = scaler.transform(scaled_data)
    
    return scaled_data, mean_values, std_values, scaler


def get_malware_dataframe(load_pickle=True):

    if load_pickle:
        df_total = pd.read_pickle('/home/ec2-user/data/df_total.pkl')
    else:
        df_train, X_train = load_data(data_set_type='train')
        df_test,  X_test  = load_data(data_set_type='test')
        # print("df_train.head():\n")
        # print(df_train.head())
        # ic(X_train.shape);
        # print("\n")
        # print("\ndf_test.head():\n")
        # print(df_test.head())
        # ic(X_test.shape);
        
        X_train_scaled, mean_values_train, std_values_train, scaler_train = scale_data(X_train)
        X_test_scaled, _, _, _ = scale_data(X_test, mean_values=mean_values_train, std_values=std_values_train, scaler=scaler_train)    
        
        df_a = pd.DataFrame(X_train_scaled)
        df_a = df_a.add_prefix('feat_')
        df_a['data_set_type'] = 'train'
        df_a['avclass'] = df_train['avclass']
        
        df_b = pd.DataFrame(X_test_scaled)
        df_b = df_b.add_prefix('feat_')
        df_b['data_set_type'] = 'test'
        df_b['avclass'] = df_test['avclass']
        
        df_total = pd.concat([df_a, df_b]).reset_index(drop=True)
        # print(df_total)
        df_total.to_pickle('/home/ec2-user/data/df_total.pkl')
    
        # print("df_total:\n")
        # print(df_total)
    
    return df_total


def get_train_test_data(df, data_set_type='train', families=None, encoder=None):
    
    df_clean = df[df['data_set_type'] == data_set_type]
    if families is not None:
        df_clean = df_clean[df_clean['avclass'].isin(families)]
    
    X = df_clean.drop(columns=['data_set_type', 'avclass']).to_numpy()
    
    y_label = list(df_clean['avclass'].values)
    
    mapping_dict = {}
    if encoder is None:
        encoder = ce.one_hot.OneHotEncoder(cols=['avclass'])
        df_trans = encoder.fit_transform(df_clean)
        
        # Create a mapping dictionary
        avclass_columns = [col for col in df_trans if col.startswith('avclass')]
        for index, row in df_clean.iterrows():
            original_category = row['avclass']
            one_hot_vector = list(df_trans.loc[index, avclass_columns])
            mapping_dict[original_category] = one_hot_vector
    else:
        df_trans = encoder.transform(df_clean)
    #print(df_trans)
    
    avclass_columns = [col for col in df_trans if col.startswith('avclass')]
    y = df_trans[avclass_columns].to_numpy()
    
    families = set(df_clean['avclass'].values)
    #ic(len(y_label));
    #ic(len(y));
    
    return X, y, y_label, sorted(families), encoder, mapping_dict


def prepare_data(verbose=False, do_class_label_encoding=False, nrof_novel_families=0):

    if verbose:
        print("Loading scaled malware data")
    df_total = get_malware_dataframe(load_pickle=True)
    #print(df_total)
    most_common_train_avclass = sorted(list(set(df_total[df_total['data_set_type'] == 'train']['avclass'].values)))
    most_common_test_avclass  = sorted(list(set(df_total[df_total['data_set_type'] == 'test']['avclass'].values)))
    # ic(most_common_train_avclass);
    # ic(most_common_test_avclass);
    
    X_train_prep, y_train_prep, y_train_labels, families, hot_encoder, mapping_dict = get_train_test_data(df=df_total, data_set_type='train')
    
    #test_families = ['adposhel',
    #                 'emotet',
    #                 'fareit',
    #                 'installmonster',
    #                 'ramnit',
    #                 'sality',
    #                 'vtflooder',
    #                 'xtrat',
    #                 'zbot',
    #                 'zusy']
    test_families = most_common_train_avclass

    # pick a few novel families for testing
    #
    novel_families = set(most_common_test_avclass).difference(set(most_common_train_avclass))
    random.seed(42)
    novel_families = random.sample(novel_families, nrof_novel_families)
    test_families = set(test_families)
    test_families.update(novel_families)
    test_families = list(test_families)
    
    X_test_prep, y_test_prep, y_test_labels, families, _, _ = get_train_test_data(df=df_total, data_set_type='test', families=test_families, encoder=hot_encoder)
    
    if verbose:
        print("\nEvaluation based on the following families:\n")
        train_families = most_common_train_avclass
        ic(train_families);
        test_families = families
        ic(test_families);
        
        novel_families = sorted(list(set(families).difference(set(most_common_train_avclass))))
        ic(novel_families);

    if do_class_label_encoding:
        return X_train_prep, y_train_prep, X_test_prep, y_test_prep
    else:
        return X_train_prep, y_train_labels, X_test_prep, y_test_labels


def eval_xgboost(X_train, y_train, X_test, y_test, verbose=False):
    
    benchmark_clf = xgb.XGBClassifier(tree_method="hist", random_state=0)
    # print("X_train:\n")
    # print(X_train)
    # print("\n\ny_train:\n")
    # print(y_train)
    benchmark_clf.fit(X_train, y_train)

    if verbose:
        ic(X_test.shape)
    
    y_pred = benchmark_clf.predict(X_test)

    if verbose:
        print("---------------------------------------------------")
        print("Evaluating XGBoost\n")
        print("Precision score:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
        print("Recall score:   ", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        print("---------------------------------------------------")


def xgboost_baseline(verbose=False):

    for nrof_novel_families in range(5):
        print("***************************************************************************************")
        print("*")
        print("* Number of novel families:", nrof_novel_families)
        print("*")
        X_train, y_train, X_test, y_test = prepare_data(verbose=False, do_class_label_encoding=True, nrof_novel_families=nrof_novel_families)
        # ic(X_train.shape);
        # ic(y_train.shape);
        # ic(X_test.shape);
        # ic(y_test.shape);
        
        eval_xgboost(X_train, y_train, X_test, y_test, verbose)
        print("\n")

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

def make_unknown(my_list, num_elements_to_change, new_element=-1, seed=42):
    copy_list = my_list.copy()
    random.seed(seed)
    indices_to_change = random.sample(range(len(copy_list)), num_elements_to_change)

    change_dict = dict()
    for index in indices_to_change:
        change_dict[index] = my_list[index]
        copy_list[index] = new_element
    
    od = OrderedDict(sorted(change_dict.items()))
#    for key, val in od.items():
#        print("UNKOWN", key, val)
    return copy_list, od


def make_unknown_labels(y_tot_labels, percent=1):
    num_elements_to_change = int(len(y_tot_labels) * (percent/100))
    new_labels, change_dict = make_unknown(my_list=y_tot_labels, num_elements_to_change=num_elements_to_change)
    return new_labels, change_dict


def change_elems(my_list, num_elements_to_change, new_val=-1, seed=42):
    random.seed(SEED)
    indices_to_change = random.sample(range(len(my_list)), num_elements_to_change)
    for idx in indices_to_change:
        my_list[idx] = new_val
    return my_list


call_nbr = 0
def NMFk_test(X=np.zeros((0,0)), get_y=False, test_paper_mode=False):
    '''
    This method is used to test the basic functionality of HNMFk_Classifier.
    
    If the flag get_y is set, it returns values for X and y.
    If the flag test_paper_mode is set, it returns values from the LANL
    research paper (Fig. 4).
    '''
    np.random.seed(5)
    
    N = X.shape[0]
    if N == 0:
        N = 20
    W = np.random.uniform(0.1, 1.0, (N, 4))*100//1/100
    
    X = np.zeros((N, 3))
    for row in range(N):
        for col in range(3):
            X[row, col] = int(row*10 + col)
    
    if get_y:
        if test_paper_mode:
            y = ['a', -1, 'b', 'b', 'a', -1, -1, -1, -1]
            return X, y
        else:
            y = N*['a']
            y = change_elems(my_list=y, num_elements_to_change=int(np.floor(N/2)), new_val='b')
            y = change_elems(my_list=y, num_elements_to_change=4)
            return X, y
    else:
        if test_paper_mode:
            global call_nbr
            call_nbr += 1
            if call_nbr == 1:
                W = np.array([[0.5, 0.1,  0.1,  0.3],
                              [0.1, 0.3,  0.4,  0.2],
                              [0.9, 0.05, 0.03, 0.02],
                              [0.1, 0.2,  0.2,  0.5],
                              [0.2, 0.3,  0.25, 0.25],
                              [0.1, 0.6,  0.2,  0.1],
                              [0.2, 0.2,  0.4,  0.2],
                              [0.8, 0.1,  0.05, 0.05],
                              [0.2, 0.05, 0.05, 0.7]])
                H = np.zeros((4,3))
                k_opt = 4

                return W  #, H, k_opt

            elif call_nbr == 2:
                W = np.array([[0.8, 0.2],
                              [0.1, 0.9],
                              [0.4, 0.6]])
                H = np.zeros((2, 3))
                k_opt = 2

            return W
        
        return W

def NMFk(X, verbose=True, data_path='/home/ec2-user/data/malware'):

    if verbose:
        print("Saving matrix X.")
        ic(X.shape)
    savemat(os.path.join(data_path, 'X.mat'), {'X': X})
    
    #runner = pyDNMFk_Runner(itr=100, init='nnsvd', verbose=False,
    #                        norm='fro', method='mu', precision=np.float32,
    #                        checkpoint=False, sill_thr=0.6)  #, process="pyDNMFk")
    #
    #data_path = os.path.join(data_path, '')
    #results = runner.run(grid=[4,1], fpath=data_path,
    #                     fname='X', ftype='mat', results_path='results/',
    #                     k_range=[1,10], step_k=1)
    command = f"""mpiexec -n 4 python factor.py"""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(result)
    W_tot = loadmat(os.path.join(data_path, 'W_0.mat'))['W']
    for rank in range(1,4):
        filename = 'W_' + str(rank) + '.mat'
        W = loadmat(os.path.join(data_path, filename))['W']
        W_tot = np.vstack((W_tot, W))
    H = loadmat(os.path.join(data_path, 'H_0.mat'))['H']
    k_opt = W.shape[1]
    return W_tot, H, k_opt


Classification = namedtuple('Classification', 'predictions abstaining')
Signature = namedtuple('Signature', 'vector index label')
class HNMFk_Classifier():
    '''
    Semi-supervised hierarchical classifier wrapper to the NMFk algorithm.
    NMFk integrates classical Non-negative Matrix Factorization (NMF)-
    minimization with custom clustering and Silhouette statistics.
    '''
    def __init__(self, X, y, test=False, cluster_uniformity_threshold=1.0, sig_arch_filepath='/home/ec2-user/data/malware/signature_archive.pkl'):
        self.X = X
        self.y = y.copy()
        self.predictions = dict()
        self.abstaining = list()
        self.signature_archive = list()
        self.cluster_uniformity_threshold = cluster_uniformity_threshold
        self.sig_arch_filepath = sig_arch_filepath
        self.test = test
    
    def get_idx_label_dict(self, y):
        mapping = dict()
        for idx, label in enumerate(y):
            mapping[idx] = label
        return mapping
       
    def W_clustering(self, W):
        k_opt = W.shape[1]
        clusters = {i: [] for i in range(k_opt)}
        for sample_idx in range(W.shape[0]):
            cluster_label = np.argmax(W[sample_idx,:])
            clusters[cluster_label] += [(sample_idx, self.idx_label_dict[sample_idx])]
        return clusters
    
    def HNMFk(self, X, y, index_mapping=None):
        
        self.idx_label_dict = self.get_idx_label_dict(y)
        
        if not index_mapping:
            index_mapping = {idx: idx for idx in range(X.shape[0])}
        
        if self.test:
            W = NMFk_test(X, test_paper_mode=True)
        else:
            W, H, k_opt = NMFk(X)
            print("W.shape:", W.shape)
            print("H.shape:", H.shape)
            print("k_opt:", k_opt)
        
        clusters = self.W_clustering(W)
        for cluster_idx, cluster in clusters.items():
            if len(cluster)>0:
                labels = [elem[1] for elem in cluster]
                known_labels = [label for label in labels if label != -1]
                
                number_of_known_samples = len(known_labels)  #sum(x == -1 for x in labels)
                number_of_unknown_samples = len(labels) - number_of_known_samples

                if number_of_unknown_samples == 0:  # no unknown samples to make prediction
                    continue

                if number_of_known_samples == 0:  # abstaining prediction
                    sample_idxes = [elem[0] for elem in cluster]
                    true_sample_idxes = [index_mapping[idx] for idx in sample_idxes]
                    self.abstaining.extend(true_sample_idxes)
                    continue

                classify_label, max_class_known = Counter(known_labels).most_common(1)[0]
                cluster_uniformity_score = max_class_known / number_of_known_samples

                if cluster_uniformity_score < self.cluster_uniformity_threshold:
                    sample_idxes = [elem[0] for elem in cluster]
                    true_sample_idxes = [index_mapping[idx] for idx in sample_idxes]
                    X_new = X[sample_idxes,:]
                    y_new = [y[i] for i in sample_idxes]
                    index_mapping_new = {idx: true_idx for idx, true_idx in enumerate(true_sample_idxes)}
                    if X_new.shape[0] > X_new.shape[1]:
                        self.HNMFk(X_new, y_new, index_mapping_new)

                else:
                    unknown_sample_idxes = [elem[0] for elem in cluster if elem[1] == -1]
                    for idx in unknown_sample_idxes:
                        true_idx = index_mapping[idx]
                        self.y[true_idx] = classify_label
                        self.predictions[true_idx] = classify_label
                    signature = H[cluster_idx, :]
                    self.signature_archive.append((Signature(signature, cluster_idx, classify_label)))
    
    def predict(self):
        self.HNMFk(self.X, self.y)
        with open(self.sig_arch_filepath, 'wb') as f:
            pickle.dump(self.signature_archive, f)
        return Classification(dict(sorted(self.predictions.items())), self.abstaining)


def run_doctest():
    import doctest
    doctest.testmod()
    print("Doctests passed.")


def NMFk_doctest():
    """
    >>> X, y = NMFk_test(get_y=True, test_paper_mode=True)
    >>> classifier = HNMFk_Classifier(X, y, test=True)
    >>> pred = classifier.predict()
    >>> print(pred)
    Classification(predictions={5: 'a', 8: 'b'}, abstaining=[1, 6])
    """


def collect_stats(truth, pred, change_dict):
    print("DETAILS:")
    nrof_correct = 0
    for k, v in pred.predictions.items():
        pred_is_correct = (v == truth[k])
        if not pred_is_correct:
            print(f"""sample {k} is erroneously predicted as '{v}', while the correct label is '{truth[k]}'""")
        else:
            nrof_correct += 1

    print("\nSUMMARY:")
    print(f"""{len(pred.predictions)} of {len(change_dict)} unknown samples are predicted, of which {nrof_correct} are correct""")
    print(f"""{len(change_dict) - len(pred.predictions)} of {len(change_dict)} unknown samples are not predicted""")
    print(f"""{len(pred.abstaining)} of {len(change_dict)} unknown samples are abstained from prediction""")
    print(f"""{len(truth)} samples in total""")


def MalwareDNA_train(verbose):
    if verbose:
        print(f"""Running {inspect.currentframe().f_code.co_name}""")
    
    X_train, y_train, X_test, y_test = prepare_data(verbose=verbose)
    
    y_tot_labels = y_train
    y_new_labels, change_dict = make_unknown_labels(y_tot_labels)
    
    classifier = HNMFk_Classifier(X=X_train, y=y_new_labels, cluster_uniformity_threshold=0.85)
    pred = classifier.predict()
    #print(pred)
    #print(change_labels)
    collect_stats(truth=y_tot_labels, pred=pred, change_dict=change_dict)


def MalwareDNA_inference(verbose, sig_arch_filepath='/home/ec2-user/data/malware/signature_archive.pkl'):
    if verbose:
        print(f"""Running {inspect.currentframe().f_code.co_name}""")
    
    with open(sig_arch_filepath, 'rb') as f:
        signature_archive = pickle.load(f)  # signature_archive

    H_selected = signature_archive[0].vector
    for signature in signature_archive[1:]:
        H_selected = np.vstack((H_selected, signature.vector))
    H_selected = H_selected.T
    
    selected_columns = list()
    corresponding_labels = list()
    for signature in signature_archive:
        selected_columns.append(signature.index)
        corresponding_labels.append(signature.label)
    
    X_train, y_train, X_test, y_test = prepare_data(verbose=verbose)
    
    y_tot_labels = y_train
    y_new_labels, change_dict = make_unknown_labels(y_tot_labels)
    
    nrof_correct = 0
    nrof_error = 0
#    ic(H_selected.shape)
#    ic(len(change_dict))
#    print("corresponding_labels", corresponding_labels)
    nrof_signatures = len(corresponding_labels)
    for row_idx, true_label in change_dict.items():
        x_sample = X_train[row_idx, :]
#        ic(x_sample.shape);
        coefficients, _ = nnls(H_selected, x_sample)  # Solve NNLS problem: W_selected * coefficients ≈ x_row
#        ic(len(coefficients));
#        print("coefficients", coefficients)
#        print("true_label", true_label)
#        ic(corresponding_labels);
#        ic(true_label);
#       pred_label = corresponding_labels[np.argmax(coefficients)]
#        cosine_sim_score = cosine_similarity([coefficients], H_selected)
#        ic(len(cosine_sim_score[0]));
#       print(cosine_sim_score[0])

        
        for idx in range(nrof_signatures):
            label_array = np.zeros((1, nrof_signatures))
            label_array[(0, idx)] = 1
#            ic(label_array.shape);
#            ic(coefficients.shape);
            if cosine_similarity([coefficients], label_array) > 0.9:
                pred_label = corresponding_labels[idx]
#                print(pred_label, true_label)
                if pred_label != true_label:
                    nrof_error += 1
                else:
                    nrof_correct += 1


#       max_cosine_sim_score = np.max(cosine_sim_score)
#       if max_cosine_sim_score > 0.7:
#           pred_label = corresponding_labels[np.argmax(cosine_sim_score)]
#       if pred_label != true_label:
#           nrof_error += 1
#       else:
#           nrof_correct += 1
    ic(nrof_correct);
    ic(nrof_error);


#from scipy.optimize import nnls
#
## Assuming X_row is a specific row from X
#for i in range(X.shape[0]):
#    x_row = X[i, :]
#    
#    # Solve NNLS problem: W_selected * coefficients ≈ x_row
#    coefficients, _ = nnls(W_selected, x_row)#
#
#    # Use coefficients to approximate the row
#    x_approximated = np.dot(W_selected, coefficients)
#
#    # Now x_approximated is the approximation of x_row using the selected signatures


def main():
    
    args, parser = parse_args()
    do_train     = args.options.train
    do_inference = args.options.inference
    verbose      = args.options.verbose
    
    if do_train:
        MalwareDNA_train(verbose)
    
    elif do_inference:
        MalwareDNA_inference(verbose)

    else:
        parser.print_help()


if __name__ == '__main__':
    #xgboost_baseline(verbose=True)
    #run_doctest()
    main()

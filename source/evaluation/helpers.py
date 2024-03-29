import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
def get_tra_zero_datastream(dataset):
    X = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep=" ", header=None)
    X = X.values
    n = X.shape[0]
    feat = X.shape[1]
    perm = np.arange(n)
    np.random.seed(1)
    np.random.shuffle(perm)
    perm = np.array(perm)
    X = X[perm]
    X = X.tolist()
    star_row = 0
    end_column = 0
    X_trapezoid = []
    X_masked = []
    X_zeros = []
    for i in range(5):
        end_row = star_row + math.ceil(n / 5)
        if end_row > n: end_row = n
        end_column = end_column + math.ceil(feat / 5)
        if end_column > feat: end_column = feat
        for j in range(star_row, end_row):
            row_1 = X[j][0:end_column]
            row_2 = row_1 + [np.nan] * (feat - end_column)
            row_3 = row_1 + [0] * (feat - end_column)

            X_trapezoid.append(row_1)
            X_masked.append(row_2)
            X_zeros.append(row_3)
        star_row = end_row

    path = "../dataset/MaskData/" + dataset
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path + "/X.txt", np.array(X_masked))
    np.savetxt(path + "/X_trapezoid_zeros.txt", np.array(X_zeros))
    file = open(path + "/X_trapezoid.txt", 'w')
    for fp in X_trapezoid:
        file.write(str(fp))
        file.write('\n')
    file.close()

def chack_Nan(X_masked,n):
    for i in range(n):
        X_masked[i] = X_masked[i].strip()
        X_masked[i] = X_masked[i].strip("[]")
        X_masked[i] = X_masked[i].split(",")
        X_masked[i] = list(map(float, X_masked[i]))
        narry = np.array(X_masked[i])
        where_are_nan = np.isnan(narry)
        narry[where_are_nan] = 0
        X_masked[i] = narry.tolist()
    return X_masked

def get_cont_indices(X):
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices

def cont_to_binary(x):
    # make the cutoff a random sample and ensure at least 10% are in each class
    while True:
        cutoff = np.random.choice(x)
        if len(x[x < cutoff]) > 0.1*len(x) and len(x[x < cutoff]) < 0.9*len(x):
            break
    return (x > cutoff).astype(int)

def cont_to_ord(x, k):
    # make the cutoffs based on the quantiles
    std_dev = np.std(x)
    cuttoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    ords = np.zeros(len(x))
    for cuttoff in cuttoffs:
        ords += (x > cuttoff).astype(int)
    return ords.astype(int)

def get_mae(x_imp, x_true, x_obs=None):
    if x_obs is not None:
        loc = np.isnan(x_obs)
        imp = x_imp[loc]
        val = x_true[loc]
        return np.mean(np.abs(imp - val))
    else:
        return np.mean(np.abs(x_imp - x_true))

def get_smae(x_imp, x_true, x_obs, Med=None, per_type=False, cont_loc=None, bin_loc=None, ord_loc=None):
    error = np.zeros((x_obs.shape[1],2))
    for i, col in enumerate(x_obs.T):
        test = np.bitwise_and(~np.isnan(x_true[:,i]), np.isnan(col))
        if np.sum(test) == 0:
            error[i,0] = np.nan
            error[i,1] = np.nan
            continue
        col_nonan = col[~np.isnan(col)]
        x_true_col = x_true[test,i]
        x_imp_col = x_imp[test,i]
        if Med is not None:
            median = Med[i]
        else:
            median = np.median(col_nonan)
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        error[i,0] = np.sum(diff)
        error[i,1]= np.sum(med_diff)
    if per_type:
        if not cont_loc:
            cont_loc = [True] * 5 + [False] * 10
        if not bin_loc:
            bin_loc = [False] * 5 + [True] * 5 + [False] * 5
        if not ord_loc:
            ord_loc = [False] * 10 + [True] * 5
        loc = [cont_loc, bin_loc, ord_loc]
        scaled_diffs = np.zeros(3)
        for j in range(3):
            scaled_diffs[j] = np.sum(error[loc[j],0])/np.sum(error[loc[j],1])
    else:
        scaled_diffs = error[:,0] / error[:,1]
    return scaled_diffs

def get_smae_per_type(x_imp, x_true, x_obs, cont_loc=None, bin_loc=None, ord_loc=None):
    if not cont_loc:
        cont_loc = [True] * 5 + [False] * 10
    if not bin_loc:
        bin_loc = [False] * 5 + [True] * 5 + [False] * 5
    if not ord_loc:
        ord_loc = [False] * 10 + [True] * 5
    loc = [cont_loc, bin_loc, ord_loc]
    scaled_diffs = np.zeros(3)
    for j in range(3):
        missing = np.isnan(x_obs[:,loc[j]])
        med = np.median(x_obs[:,loc[j]][~missing])
        diff = np.abs(x_imp[:,loc[j]][missing] - x_true[:,loc[j]][missing])
        med_diff = np.abs(med - x_true[:,loc[j]][missing])
        scaled_diffs[j] = np.sum(diff)/np.sum(med_diff)
    return scaled_diffs

def get_rmse(x_imp, x_true, relative=False):
    diff = x_imp - x_true
    mse = np.mean(diff**2.0, axis=0)
    rmse = np.sqrt(mse)
    return rmse if not relative else rmse/np.sqrt(np.mean(x_true**2))

def get_relative_rmse(x_imp, x_true, x_obs):
    loc = np.isnan(x_obs)
    imp = x_imp[loc]
    val = x_true[loc]
    return get_scaled_error(imp, val)

def get_scaled_error(sigma_imp, sigma):
    return np.linalg.norm(sigma - sigma_imp) / np.linalg.norm(sigma)

def mask_types(X, mask_num, seed):
    # X_masked = np.copy(X)
    # mask_indices = []
    # num_rows = X_masked.shape[0]
    # num_cols = X_masked.shape[1]
    # # mask = 0.5
    # x_masked = random.sample(range(1, X_masked.shape[1]), int(X_masked.shape[1] * furter_mask))
    # # print('x_masked:',x_masked)
    # for i in range(num_rows):
    #     np.random.seed(seed*num_rows-i) # uncertain if this is necessary
    #     for idx in x_masked:
    #
    #       X_masked[i,idx]=np.nan
    #       mask_indices.append((i,idx))
    #       # print(len(mask_indices))
    # return X_masked

    X_masked = np.copy(X)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i) # uncertain if this is necessary
        for j in range(num_cols//2):
            rand_idx=np.random.choice(2,mask_num,False)
            for idx in rand_idx:
                # print('1',idx+3*j)

                X_masked[i,idx+2*j]=np.nan
                mask_indices.append((i, idx+2*j))
                # print(len(mask_indices))
    return X_masked

def mask(X, mask_fraction, seed=0, verbose=False):
    complete = False
    count = 0
    X_masked = np.copy(X)
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    while not complete:
        np.random.seed(seed)
        if (verbose): print(seed)
        mask_indices = obs_indices[np.random.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
        for i,j in mask_indices:
            X_masked[i,j] = np.nan
        complete = True
        for row in X_masked:
            if len(row[~np.isnan(row)]) == 0:
                seed += 1
                count += 1
                complete = False
                X_masked = np.copy(X)
                break
        if count == 50:
            raise ValueError("Failure in Masking data without empty rows")
    return X_masked, mask_indices, seed

def mask_per_row(X, seed=0, size=1):
    X_masked = np.copy(X)
    n,p = X.shape
    for i in range(n):
        np.random.seed(seed*n+i)
        rand_idx = np.random.choice(p, size)
        X_masked[i,rand_idx] = np.nan
    return X_masked

def _project_to_correlation(covariance):
        D = np.diagonal(covariance)
        D_neg_half = np.diag(1.0/np.sqrt(D))
        return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_sigma(seed):
    np.random.seed(seed)
    W = np.random.normal(size=(18, 18))
    covariance = np.matmul(W, W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def continuous2ordinal(x, k = 2, cutoff = None):
    q = np.quantile(x, (0.05,0.95))
    if k == 2:
        if cutoff is None:
            # random cuttoff from the data between the 5th and 95th percentile
            cutoff = np.random.choice(x[(x > q[0])*(x < q[1])])
        x = (x >= cutoff).astype(int)
    else:
        if cutoff is None:
            std_dev = np.std(x)
            min_cutoff = np.min(x) - 0.1 * std_dev
            cutoff = np.sort(np.random.choice(x[(x > q[0])*(x < q[1])], k-1, False))
            max_cutoff = np.max(x) + 0.1 * std_dev
            cuttoff = np.hstack((min_cutoff, cutoff, max_cutoff))
        x = np.digitize(x, cuttoff)
    return x

def grassman_dist(A,B):
    U1, d1, _ = np.linalg.svd(A, full_matrices = False)
    U2, d2, _ = np.linalg.svd(B, full_matrices = False)
    _, d,_    = np.linalg.svd(np.dot(U1.T, U2))
    theta     = np.arccos(d)
    return np.linalg.norm(theta), np.linalg.norm(d1-d2)

def get_tra_hyperparameter(dataset):
    decay_choices             = {"ionosphere": 4,     "wbc": 1,     "wdbc": 3,     "german": 2,      "diabetes": 2,    "credit":4,         "australian":4,     "svmguide3":4,    "kr_vs_kp":4,     "magic04":4,    "wpbc": 4     ,"a8a": 4     , "splice":4    , "dna": 4    , "Stream1":4}
    contribute_error_rates    = {"ionosphere": 0.01,  "wbc": 0.005, "wdbc": 0,     "german": 0.005,  "diabetes": 0,    "credit":0.01,      "australian":0.02,  "svmguide3":0.01, "kr_vs_kp":0.01,  "magic04":0.01, "wpbc": 0.01  ,"a8a": 0.01  , "splice":0.01 , "dna": 0.01 , "Stream1":0.01}
    window_size_denominators  = {"ionosphere": 2,     "wbc": 2,     "wdbc": 2,     "german": 8,      "diabetes": 2,    "credit":2,         "australian":2,     "svmguide3":2,    "kr_vs_kp":2,     "magic04":2,    "wpbc": 8     ,"a8a": 2     , "splice":2    , "dna": 4    , "Stream1":4}
    batch_size_denominators   = {"ionosphere": 20,    "wbc": 8,     "wdbc": 2,     "german": 8,      "diabetes": 8,    "credit":4,         "australian": 10,   "svmguide3": 8,   "kr_vs_kp": 8,    "magic04":8,    "wpbc": 8     ,"a8a": 4     , "splice":8    , "dna": 8    , "Stream1":8}
    shuffles                  = {"ionosphere": False, "wbc": True,  "wdbc": True,  "german": False,  "diabetes": True, "credit":True,      "australian":False, "svmguide3":True, "kr_vs_kp":False, "magic04":True, "wpbc": False ,"a8a": False , "splice":True , "dna": False, "Stream1":False}

    decay_coef_change=0
    batch_size_denominator = batch_size_denominators[dataset]
    contribute_error_rate=contribute_error_rates[dataset]
    window_size_denominator=window_size_denominators[dataset]
    shuffle=shuffles[dataset]
    decay_choice=decay_choices[dataset]
    return contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change, decay_choice, shuffle

def get_cap_hyperparameter(dataset):
    decay_choices            = {"ionosphere": 4,     "wbc": 2,    "wdbc": 0,     "german": 3,     "diabetes": 2,     "credit": 4,         "australian":4,      "wpbc": 4,     "kr_vs_kp":4,     "svmguide3":4,    "magic04":4,    "imdb":2,     "synthetic":4     ,"a8a": 4     , "splice":4    , "dna": 4     , "hapt": 4     , "Stream1": 4   ,"steamData":4}
    contribute_error_rates   = {"ionosphere": 0.02,  "wbc": 0.02, "wdbc": 0.02,  "german": 0.005, "diabetes": 0.05,  "credit": 0.01,      "australian":0.01,   "wpbc": 0.01,  "kr_vs_kp":0.01,  "svmguide3":0.01, "magic04":0.01, "imdb":0.01,  "synthetic":0.01  ,"a8a": 0.01  , "splice":0.01 , "dna": 0.01  , "hapt": 0.01  , "Stream1": 0.01 ,"steamData":0.01}
    window_size_denominators = {"ionosphere": 2,     "wbc": 2,    "wdbc": 2,     "german": 2,     "diabetes": 2,     "credit": 2,         "australian":2   ,   "wpbc": 8,     "kr_vs_kp":2,     "svmguide3":2,    "magic04":2,    "imdb":2,     "synthetic":2     ,"a8a": 2     , "splice":4    , "dna": 4     , "hapt": 4     , "Stream1": 4 ,"steamData":2}
    batch_size_denominators  = {"ionosphere": 20,    "wbc": 8,    "wdbc": 8,     "german": 8,     "diabetes": 8,     "credit": 4,         "australian": 10  ,  "wpbc": 8,     "kr_vs_kp":8,     "svmguide3":8,    "magic04":2,    "imdb":8,     "synthetic":8    ,"a8a": 8     , "splice":4    , "dna": 8     , "hapt": 8     , "Stream1": 8 ,"steamData":8}
    shuffles                 = {"ionosphere": True,  "wbc": False,"wdbc": True,  "german": False, "diabetes": True,  "credit": True,      "australian":False , "wpbc": False, "kr_vs_kp":False, "svmguide3":True, "magic04":True, "imdb":False, "synthetic":False ,"a8a": False , "splice":True , "dna": False , "hapt": False , "Stream1":False ,"steamData":False}

    batch_size_denominator=batch_size_denominators[dataset]
    decay_coef_change=0
    contribute_error_rate=contribute_error_rates[dataset]
    window_size_denominator=window_size_denominators[dataset]
    shuffle=shuffles[dataset]
    decay_choice=decay_choices[dataset]

    return contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,shuffle

def Cumulative_error_rate_semi_ensemble(predict_label_train_x_ensemble, predict_label_train_z_ensemble, Y_label):
    n = len(predict_label_train_x_ensemble)
    errors_x = []
    errors_z = []

    for i in range(n):
        y = Y_label[i]
        x = predict_label_train_x_ensemble[i]
        z = predict_label_train_z_ensemble[i]
        error_x = [int(np.abs(y - x) > 0.5)]
        error_z = [int(np.abs(y - z) > 0.5)]
        errors_x.append(error_x)
        errors_z.append(error_z)

    X_CER_ensemble = np.cumsum(errors_x) / (np.arange(len(errors_x)) + 1.0)
    Z_CER_ensemble = np.cumsum(errors_z) / (np.arange(len(errors_z)) + 1.0)

    l = len(X_CER_ensemble)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER of X & Z")
    x = range(l)

    X_CER_ensemble_line, = plt.plot(x, X_CER_ensemble, color='green', linestyle="--")  # the error of z_imp
    Z_CER_ensemble_line, = plt.plot(x, Z_CER_ensemble, color='blue',  linestyle="-")  # the error of x_zero

    plt.legend(handles=[X_CER_ensemble_line, Z_CER_ensemble_line], labels=["X_CER_ensemble_line", "Z_CER_ensemble_line"])

    plt.title(
        "The Cumulative error rate(CER) of X_CER_ensemble_line, Z_CER_ensemble_line")

    plt.show()

def Cumulative_error_rate_semi(predict_label_train_x_ensemble, predict_label_train_z_ensemble, Y_label_fill_x,
                               Y_label_fill_z, Y_label, dataset):
    n = len(predict_label_train_x_ensemble)
    errors_x = []
    errors_z = []
    errors_x_ensemble = []
    errors_z_ensemble = []
    dataset = dataset
    for i in range(n):
        y = Y_label[i]
        x = Y_label_fill_x[i]
        z = Y_label_fill_z[i]
        x_ensemble = predict_label_train_x_ensemble[i]
        z_ensemble = predict_label_train_z_ensemble[i]

        error_x = [int(np.abs(y - x) > 0.5)]
        error_z = [int(np.abs(y - z) > 0.5)]
        error_x_ensemble = [int(np.abs(y - x_ensemble) > 0.5)]
        error_z_ensemble = [int(np.abs(y - z_ensemble) > 0.5)]

        errors_x.append(error_x)
        errors_z.append(error_z)
        errors_x_ensemble.append(error_x_ensemble)
        errors_z_ensemble.append(error_z_ensemble)

    X_CER = np.cumsum(errors_x) / (np.arange(len(errors_x)) + 1.0)
    Z_CER = np.cumsum(errors_z) / (np.arange(len(errors_z)) + 1.0)
    X_CER_ensemble = np.cumsum(errors_x_ensemble) / (np.arange(len(errors_x_ensemble)) + 1.0)
    Z_CER_ensemble = np.cumsum(errors_z_ensemble) / (np.arange(len(errors_z_ensemble)) + 1.0)

    l = len(X_CER_ensemble)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER of X & Z")
    x = range(l)

    X_CER_line, = plt.plot(x, X_CER, color='pink', linestyle="-.")
    Z_CER_line, = plt.plot(x, Z_CER, color='black', linestyle=":")

    X_CER_ensemble_line, = plt.plot(x, X_CER_ensemble, color='green', linestyle="--")
    Z_CER_ensemble_line, = plt.plot(x, Z_CER_ensemble, color='blue',  linestyle="-")

    plt.legend(handles=[X_CER_ensemble_line, Z_CER_ensemble_line, X_CER_line, Z_CER_line],
               labels=["X_CER_ensemble_line", "Z_CER_ensemble_line", "X_CER_line", "Z_CER_line"])

    plt.title(dataset + "The Cumulative error rate(CER) of X_CER_ensemble_line, Z_CER_ensemble_line, X_CER_line, Z_CER_line")
    # plt.savefig("E:/俺的文档/研二工作/Auto_local_density_peak/Auto_local_density_peak/experiment/CER/australian/error.jpg")

    # plt.savefig("G:/俺的文档/研二工作/Auto_local_density_peak/Auto_local_density_peak/result/steamData/error.jpg")
    plt.show()

# Use to magic04
def shuffle_dataset_1(data):
    da1index = data[data.values == 1]
    da1index = da1index.index
    da1index = da1index.tolist()

    da0index = data[data.values == 0]
    da0index = da0index.index
    da0index = da0index.tolist()

    shufflearray = []
    j = 0
    for i in range(len(da1index)):
        shufflearray.append(da1index[i])
        if i % 3 == 0:
            shufflearray.append(da0index[j])
            j += 1

    for n in range(j, len(da0index)):
        shufflearray.append(da0index[n])

    shufflearray = np.array(shufflearray)
    shufflearray.flatten()
    return shufflearray
def shuffle_dataset(data):
    da1index = data[data.values == 1]
    da1index = da1index.index
    da1index = da1index.tolist()

    da0index = data[data.values == 0]
    da0index = da0index.index
    da0index = da0index.tolist()

    shufflearray = []
    j = 0
    for i in range(len(da0index)):
        shufflearray.append(da0index[i])
        if i % 2 == 0:
            shufflearray.append(da1index[j])
            j += 1
    for n in range(j, len(da1index)):
        shufflearray.append(da1index[n])

    shufflearray = np.array(shufflearray)
    shufflearray.flatten()
    return shufflearray
# Use to magic04
def shuffle_dataset_1(data):
    da1index = data[data.values == 1]
    da1index = da1index.index
    da1index = da1index.tolist()

    da0index = data[data.values == 0]
    da0index = da0index.index
    da0index = da0index.tolist()

    shufflearray = []
    j = 0
    for i in range(len(da1index)):
        shufflearray.append(da1index[i])
        if i % 3 == 0:
            shufflearray.append(da0index[j])
            j += 1

    for n in range(j, len(da0index)):
        shufflearray.append(da0index[n])

    shufflearray = np.array(shufflearray)
    shufflearray.flatten()
    return shufflearray

def shuffle_dataset_2(data):
    da1index = data[data.values == 1]
    da1index = da1index.index
    da1index = da1index.tolist()

    da0index = data[data.values == 0]
    da0index = da0index.index
    da0index = da0index.tolist()

    array = []
    j = 0
    for i in range(len(da0index)):
        array.append(da0index[i])
        if i % 4 == 0:
            array.append(da1index[j])
            j += 1
    for n in range(j, len(da1index)):
        array.append(da1index[n])

    array = np.array(array)
    array.flatten()
    return array
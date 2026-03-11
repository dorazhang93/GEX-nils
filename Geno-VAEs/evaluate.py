import numpy as np
from project import inference_all, inference_val_missing
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from config import *
import json
config, out_dir, _ = load_config()
rng = np.random.RandomState(16)

def norm(input):
    sc = StandardScaler()
    sc.fit(input)
    out = sc.transform(input)
    return out, sc.scale_, sc.mean_


# cross validation
# X = np.load(out_dir / "latent_features.npy")
# Y = np.load(config["data_params"]["data_path"] + config["data_params"]["data_name"] + "_Y_all.npy")
# Y_norm, Y_scale, Y_mean = norm(Y)
# Y_date =Y_norm[:,0]
# Y_location = Y_norm[:,2:]
# idx = np.arange(Y.shape[0])
# _,_,idx_train,idx_test = train_test_split(idx,idx,test_size=0.15,random_state=rng)

# use Sara's split
# ENCODE="EnEurasia"
# ENCODE="testEn1k"
# ENCODE="EnGlobal"
ENCODE="stageIIEn6kEurasia"
X_train = np.load(out_dir / "train_latent_features.npy")
Y_train = np.load("/home/avesta/daqu/Projects/genome/data/aDNA/final/Eurasian/diff_encoding/TPS_6535inds/TPS_eurasia_train6535_MF_Y_all.npy")
X_test = np.load(out_dir / "test_latent_features.npy")
Y_test = np.load("/home/avesta/daqu/Projects/genome/data/aDNA/final/Eurasian/same_popEurasian_encoding/TPS_test1264/TPS_eurasia_test1264_MF_Y_all.npy")
Y=np.append(Y_train,Y_test,axis=0)
Y_norm, Y_scale, Y_mean = norm(Y)
Y_date =Y_norm[:,0]
Y_location = Y_norm[:,2:]
idx_train = len(Y_train)


def eval_missing_imputation():
    sparsifies = np.linspace(0.1,1.0,5)
    mf_errors, AE_errors, miss_rates = inference_val_missing(sparsifies)
    # fig, ax = plt.subplots(1,1)
    # plt.plot(miss_rates,mf_errors,label="MSE(most frequent imputation)")
    # plt.plot(miss_rates,AE_errors,label="MSE(AE imputaion)")
    # plt.legend()
    # plt.savefig(out_dir/"imputation_error.png")
    result = {"mostFreq_error":mf_errors,"AE_error":AE_errors,"miss_rates":miss_rates}
    print(result)
    with open(out_dir / "impute_error.json","w") as f:
        f.write(json.dumps(result))
    return

def extra_eval_time_regression():
    model = RandomForestRegressor(random_state=rng)
    model.fit(X_train,Y_date[:idx_train])
    predict_test = model.predict(X_test)*Y_scale[0]+Y_mean[0]
    predict_train = model.predict(X_train)*Y_scale[0]+Y_mean[0]

    ae_test = np.abs(Y_test[:,0]-predict_test)
    ae_train = np.abs(Y_train[:,0]-predict_train)
    result = {"Date regression":{"Mean Absolute Error":{"test":{"average":ae_test.mean(), "std": ae_test.std()},
                                                        "train":{"average":ae_train.mean(), "std": ae_train.std()},}}}
    print(result)
    with open(out_dir / f"Date_regression_{ENCODE}.json", "w") as f:
        f.write(json.dumps(result))

    np.save(out_dir / f"test_date_prediction_{ENCODE}.npy", predict_test)
    return

def extra_eval_location_regression():
    model = RandomForestRegressor(random_state=rng)
    model.fit(X_train,Y_location[:idx_train,:])

    predict_test = model.predict(X_test)
    predict_train = model.predict(X_train)

    ae_test = np.sqrt((np.abs(Y_location[idx_train:,:]-predict_test)**2).sum(axis=1))
    r2_test = r2_score(Y_location[idx_train:,:],predict_test)
    ae_train = np.sqrt((np.abs(Y_location[:idx_train, :] - predict_train) ** 2).sum(axis=1))
    r2_train = r2_score(Y_location[:idx_train, :], predict_train)
    result = {"Location regression":{"Mean Absolute Error":{"test":{"average":ae_test.mean(), "std": ae_test.std()},
                                                            "train":{"average":ae_train.mean(), "std": ae_train.std()}},
                                 "r2 score":{"test":{"average":r2_test},"train":{"average":r2_train}}}}
    print(result)
    with open(out_dir / f"Location_regression_{ENCODE}.json", "w") as f:
        f.write(json.dumps(result))

    np.save(out_dir / f"test_location_prediction_{ENCODE}.npy", predict_test*Y_scale[2:]+Y_mean[2:])
    return

# def cross_val_time_regression():
#     model = RandomForestRegressor(random_state=rng)
#     cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=rng)
#     cv_results = cross_validate(model,X,Y_date,cv=cv,scoring=('r2','neg_mean_absolute_error'))
#     mae = cv_results["test_neg_mean_absolute_error"].mean() * (-Y_scale[0])
#     stdae = cv_results["test_neg_mean_absolute_error"].std() * (Y_scale[0])
#     r2 = cv_results["test_r2"].mean()
#     stdr2 = cv_results["test_r2"].std()
#     result = {"Date regression":{"Mean Absolute Error":{"average":mae, "std": stdae},
#                                  "r2 score":{"average":r2, "std": stdr2}}}
#     print(result)
#     with open(out_dir / "Date_regression.json", "w") as f:
#         f.write(json.dumps(result))
#     return
#

# def cross_val_location_regression():
#     model = RandomForestRegressor(random_state=rng)
#     cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=rng)
#     cv_results = cross_validate(model,X,Y_location,cv=cv,scoring=('r2','neg_mean_absolute_error'))
#     mae = cv_results["test_neg_mean_absolute_error"].mean() *(-1)
#     stdae = cv_results["test_neg_mean_absolute_error"].std()
#     r2 = cv_results["test_r2"].mean()
#     stdr2 = cv_results["test_r2"].std()
#     result = {"location regression":{"Mean Absolute Error":{"average":mae, "std": stdae},
#                                  "r2 score":{"average":r2, "std": stdr2}}}
#     print(result)
#     with open(out_dir / "Location_regression.json", "w") as f:
#         f.write(json.dumps(result))
#     return

if __name__=="__main__":
    # eval_missing_imputation()
    extra_eval_time_regression()
    extra_eval_location_regression()

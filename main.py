########################
# Lior Landau 201249976
# Matan Sudry 203495411
########################

import warnings
import numpy as np
import pandas as pd
import copy
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import math
from scipy import stats

warnings.filterwarnings("ignore")
STRATIFICATION_BINS = 15
THRESHOLD_S_IPW = 5
THRESHOLD_S_DR = 30


def read_data(path):
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :]
    X = pd.get_dummies(X)
    T = df['T']
    Y = df['Y']
    return df, T, Y


def Creating_threshold(T_values, T, Threshold):
    cnt = 1
    for i in T_values:
        if i < Threshold:
            T[cnt] = 0
        else:
            T[cnt] = 1
        cnt = 1 + cnt
    return T


def propensity_score_LR(df):  # calculates propensity score by GLM estimate
    X = df.iloc[:, :9]
    T = df['T']
    model = LogisticRegression()
    model.fit(X, T)
    prop = np.asarray(model.predict_proba(X))
    return prop


### need to fix below###


def S_learner(df):  # calculates ATE by S-learner
    df_copy = df.copy()
    X = df_copy.iloc[:, :-1]
    y = df_copy['Y']
    model = LinearRegression().fit(X, y)
    df_2 = df_copy[df_copy['T'] == 1]
    pred_1 = model.predict(df_2.iloc[:, :-1])
    df_2["T"] = 0
    pred_2 = model.predict(df_2.iloc[:, :-1])
    ATE = (float(sum(pred_1)) / float(len(pred_1))) - (float(sum(pred_2)) / float(len(pred_2)))
    return ATE


def T_learner(df):  # calculates ATE by T-learner
    df_copy = df.copy()
    model_1 = LinearRegression()
    model_2 = LinearRegression()
    df_1 = df_copy[df_copy["T"] == 1].drop(["T"], axis=1)
    df_2 = df_copy[df_copy["T"] == 0].drop(["T"], axis=1)
    model_1.fit(df_1.iloc[:, :-1], df_1.iloc[:, -1])
    model_2.fit(df_2.iloc[:, :-1], df_2.iloc[:, -1])
    pred_1 = model_1.predict(df_1.iloc[:, :-1])
    pred_2 = model_2.predict(df_1.iloc[:, :-1])
    ATE = (float(sum(pred_1)) / float(len(pred_1))) - (float(sum(pred_2)) / float(len(pred_2)))
    return ATE


def Matching(df):  # calculates ATE by matching
    t_1 = df[df['T'] == 1].drop(["T"], axis=1).values
    t_0 = df[df['T'] == 0].drop(["T"], axis=1).values
    t_2 = []
    for row in t_1:
        min_dist = 10000000000000
        val = 100000000000000
        for sub_row in t_0:
            dist = euclidean(row[:-1], sub_row[:-1])
            if dist < min_dist:
                min_dist = dist
                val = row[-1] - sub_row[-1]
        t_2.append(val)
    ATT = float(sum(t_2)) / float(len(t_2))
    return ATT


def stratification(data):  # calculates ATE by stratification
    ate_scores = []
    data = data.sort_values(by=['ps_1'])
    data['stratification_bin'] = pd.qcut(data['ps_1'], q=STRATIFICATION_BINS,
                                         labels=[i for i in range(1, STRATIFICATION_BINS + 1)], retbins=False)
    is_rct = 1
    for bin in range(1, STRATIFICATION_BINS + 1):
        inner_bin_data = copy.deepcopy(data[data['stratification_bin'] == bin])
        ate_scores.append(inverse_propensity_score(inner_bin_data, is_rct))
    ate = np.nanmean(ate_scores)
    return ate


def get_group_effect(data, is_treatment=1, is_rct=1):
    T = data['T']
    y = data['Y']
    ps = data['ps_1']
    group_record_size = data[data['T'] == is_treatment].shape[0]
    n = data.shape[0]
    group_effect = -1
    if (group_record_size >= THRESHOLD_S_IPW):
        if is_rct:
            if is_treatment:
                group_effect = ((T * y / 0.5).sum()) / (group_record_size)
            else:
                group_effect = ((((1 - T) * y) / 0.5).sum()) / (group_record_size)
        else:
            if is_treatment:
                group_effect = (((T * y) / ps).sum()) / n
            else:
                group_effect = ((((1 - T) * y) / (1 - ps)).sum()) / n
    return group_effect


def inverse_propensity_score(data, is_rct=1):
    treatment = 1
    control = 0
    treatment_effect = get_group_effect(data, treatment, is_rct)
    control_effect = get_group_effect(data, control, is_rct)
    if (treatment_effect != -1 and control_effect != -1):
        ate = treatment_effect - control_effect
    else:
        ate = np.nan
    return ate


######################################################### Other team project#############################################################


def propensity_score_lr_predictor(X, T):
    clf = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', multi_class='multinomial').fit(X, T)
    ps_score = np.asarray(clf.predict_proba(X))
    T_pred = clf.predict(X)
    # print('Accuracy LR: ' + str(accuracy_score(T,T_pred)))
    return ps_score


def ridge_linear_regression(adj_bin_data, with_ps=True):
    if with_ps:
        X = adj_bin_data.drop(columns=['Y', 'ps_0', 'ps_1'])
    else:
        X = adj_bin_data.drop(columns=['Y'])
    y = adj_bin_data['Y']
    ridge = Ridge(random_state=0)
    clf = ridge.fit(X, y)
    T_coef = clf.coef_[-1]
    y_pred = clf.predict(X)
    # y_pred_T = clf.predict(X[X["T"]==1])
    # y_pred_C = clf.predict(X[X["T"]==0])
    # ate = (y_pred_T-y_pred_C).mean()
    # print('RMSE: ' + str(math.sqrt(mean_squared_error(y,y_pred))))
    # t_ind,p_value = stats.ttest_ind(y,y_pred,equal_var=False)
    # print ('linear_t: ' + str (t_ind) + ' linear_p: '+ str(p_value))
    return T_coef, y_pred


def ridge_non_linear_regression(adj_bin_data):
    X = adj_bin_data.drop(columns=['Y', 'ps_0', 'ps_1'])
    y = adj_bin_data['Y']
    columns = X.columns
    ridge = Ridge(random_state=0)
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)
    clf = ridge.fit(X_poly, y)
    features_coef = dict(zip(poly.get_feature_names(columns), clf.coef_))
    T_coef_non_linear = features_coef['T']
    y_pred = clf.predict(X_poly)
    # t_ind,p_value = stats.ttest_ind(y,y_pred,equal_var=False)
    # print ('non_linear_t: ' + str (t_ind) + ' non_linear_p: '+ str(p_value))
    # print('RMSE_non_linear: ' + str(math.sqrt(mean_squared_error(y,y_pred))))
    return T_coef_non_linear, y_pred


def ridge_non_linear_regression_robust(adj_bin_data):
    X = adj_bin_data.drop(columns=['Y', 'ps_0', 'ps_1'])
    columns = X.columns
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(data=X_poly, columns=poly.get_feature_names(columns))
    X_poly_treated = X_poly[X_poly['T'] == 1]
    X_poly_control = X_poly[X_poly['T'] == 0]
    y_treated = adj_bin_data[adj_bin_data['T'] == 1]['Y']
    y_control = adj_bin_data[adj_bin_data['T'] == 0]['Y']
    ridge_treated = Ridge(random_state=0)
    ridge_control = Ridge(random_state=1)
    clf_treated = ridge_treated.fit(X_poly_treated, y_treated)
    clf_control = ridge_control.fit(X_poly_control, y_control)
    y_pred_treated = clf_treated.predict(X_poly)
    y_pred_control = clf_control.predict(X_poly)
    return y_pred_treated, y_pred_control


def calc_doubly_robust_sum(bin_data):
    n = bin_data.shape[0]
    T = bin_data['T']
    ps = bin_data['ps_1']
    treatment_prediction = bin_data['treated_prediction']
    control_prediction = bin_data['control_prediction']
    y = bin_data['imdb_score']

    left_hand_sum = ((T * y / ps) - ((T - ps) / ps) * treatment_prediction).sum()
    right_hand_sum = ((((1 - T) * y) / (1 - ps)) + ((T - ps) / (1 - ps)) * control_prediction).sum()
    sum_answer = (left_hand_sum - right_hand_sum) / n
    return sum_answer


def doubly_robust_estimator(bin_data):
    bin_data_treated = bin_data[bin_data['T'] == 1]
    bin_data_control = bin_data[bin_data['T'] == 0]
    if (bin_data_treated.shape[0] >= THRESHOLD_S_DR and bin_data_control.shape[0] >= THRESHOLD_S_DR):
        treated_predictions, control_predictions = ridge_non_linear_regression_robust(bin_data)
        bin_data['treated_prediction'] = treated_predictions
        bin_data['control_prediction'] = control_predictions
        ate = calc_doubly_robust_sum(bin_data)
    else:
        ate = np.nan
    return ate


def main():
    stratification_ate_list = list()
    path = "Final_Data.csv"
    data, T, Y = read_data(path)
    X = data.iloc[:, :9]
    T_values = data.iloc[:, 9]
    file = open("scores.txt", "w")
    for Threshold in range(10, 23):
        file.write('%d \n' % Threshold)
        T = Creating_threshold(T_values, T, Threshold)
        adj_data = pd.concat([X, T, Y], axis=1)
        prop = propensity_score_LR(adj_data)
        propensity_score = pd.DataFrame(data={'ps_0': prop[:, 0], 'ps_1': prop[:, 1]}, columns=['ps_0', 'ps_1'])
        propensity_score.index = np.arange(1, len(propensity_score) + 1)
        propensity_score.shift(2, axis=0)
        adj_data = pd.concat([X, propensity_score, T, Y], axis=1)
        IPW_measure = inverse_propensity_score(adj_data, 0)
        file.write("IPW = %0.8f\n" % IPW_measure)
        stratification_ate = (stratification(adj_data))
        file.write("Stratification = %0.8f\n" % stratification_ate)
        s_learner = S_learner(adj_data)
        file.write("S-learner = %0.8f\n" % s_learner)
        t_learner = T_learner(adj_data)
        file.write("T-learner = %0.8f\n" % t_learner)
        """matching=Matching(adj_data)
        file.write("Matching = %0.8f\n" % matching)"""
    file.close()


if __name__ == "__main__":
    main()
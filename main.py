########################
# Lior Landau
# Matan Sudry
########################

import warnings
import numpy as np
import pandas as pd
import copy
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


###Global Varaibles###
warnings.filterwarnings("ignore")
STRATIFICATION_ths = 15
THRESHOLD_S_IPW = 5
THRESHOLD_S_DR = 30

####################################################


###arranging data#####
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

####################################################

##########################methods###################

def propensity_score_LR(df): #GLM-family binomial (logistic regression)
    X = df.iloc[:, :9]
    T = df['T']
    model = LogisticRegression()
    model.fit(X, T)
    prop = np.asarray(model.predict_proba(X))
    return prop


def log_liklihood(df): #helper funnction for sensitivity
    X=df.drop(columns=['Y'])
    y = df['Y']
    X = df.iloc[:, :9]
    model = LogisticRegression()
    fit = model.fit(X, y)
    T_C = fit.coef_[0][-1]
    y_pred = model.predict(X)

    return T_C, y_pred


def S_learner(df):  # calculates ATE by S-learner
    df_copy = df.copy()
    X = df_copy.iloc[:, :-1]
    y = df_copy['Y']
    model= GaussianNB().fit(X, y)
    df_2 = df_copy[df_copy['T'] == 1]
    pred_1 = model.predict(df_2.iloc[:, :-1])
    df_2["T"] = 0
    pred_2 = model.predict(df_2.iloc[:, :-1])
    ATE = (float(sum(pred_1)) / float(len(pred_1))) - (float(sum(pred_2)) / float(len(pred_2)))
    return ATE


def T_learner(df):  # calculates ATE by T-learner
    df_copy = df.copy()
    model_1 = RandomForestClassifier()
    model_2 = RandomForestClassifier()
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




def RFC_robust(df): #helper classifier for Doubly Robust
    df_copy = df.copy()
    model_1 = RandomForestClassifier()
    model_2 = RandomForestClassifier()
    df_1 = df_copy[df_copy["T"] == 1].drop(["T"], axis=1)
    df_2 = df_copy[df_copy["T"] == 0].drop(["T"], axis=1)
    model_1.fit(df_1.iloc[:, :-1], df_1.iloc[:, -1])
    model_2.fit(df_2.iloc[:, :-1], df_2.iloc[:, -1])
    pred_1 = model_1.predict(df_1.iloc[:, :-1])
    pred_2 = model_2.predict(df_1.iloc[:, :-1])

    return pred_1, pred_2


def stratification(data): #stratification, binned based
    ate_scores = []
    data = data.sort_values(by=['ps_1'])
    data['stratification_th'] = pd.qcut(data['ps_1'], q=STRATIFICATION_ths,
                                         labels=[i for i in range(1, STRATIFICATION_ths + 1)], retbins=False)
    is_rct = 1
    for th in range(1, STRATIFICATION_ths + 1):
        inner_th_data = copy.deepcopy(data[data['stratification_th'] == th])
        ate_scores.append(inverse_propensity_score(inner_th_data, is_rct))
    ate = np.nanmean(ate_scores)
    return ate


def get_group_effect(data, is_treatment=1, is_rct=1): #manipulating data splits rule for IPW-helper function
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


def inverse_propensity_score(data, is_rct=1): # ATE by IPW
    treatment = 1
    control = 0
    treatment_effect = get_group_effect(data, treatment, is_rct)
    control_effect = get_group_effect(data, control, is_rct)
    if (treatment_effect != -1 and control_effect != -1):
        ate = treatment_effect - control_effect
    else:
        ate = np.nan
    return ate



def DR_expression(df): #Doubly Robust long expression calculation
    n = df.shape[0]
    T = df['T']
    ps = df['ps_1']
    tred_pred = df['tred_pred']
    ctrl_pred = df['ctrl_pred']
    y = df['Y']

    left_hand_sum = ((T * y / ps) - ((T - ps) / ps) * tred_pred).sum()
    right_hand_sum = ((((1 - T) * y) / (1 - ps)) + ((T - ps) / (1 - ps)) * ctrl_pred).sum()
    sum_answer = (left_hand_sum - right_hand_sum) / n
    return sum_answer


def doubly_robust(df): #Doubly Robust function-ATE
    treatement = df[df['T'] == 1]
    control = df[df['T'] == 0]
    if (treatement.shape[0] >= THRESHOLD_S_DR and control.shape[0] >= THRESHOLD_S_DR):
        tred_pred, ctrl_pred = RFC_robust(df)
        temp1=np.zeros(len(df)-len(tred_pred))
        temp2 = np.zeros(len(df) - len(ctrl_pred))
        tred_pred=np.append(tred_pred,temp1)
        ctrl_pred= np.append(ctrl_pred, temp2)

        df['tred_pred'] = tred_pred
        df['ctrl_pred'] = ctrl_pred
        ate = DR_expression(df)
    else:
        ate = np.nan
    return ate

####################################################

##########################sensitivity annalysis###################


def sensitivity_analysis(df, T_C,th):
    df['noise'] = df['Y'] - df['pred_y']
    var1 = df[df['T'] == 1]['noise'].var()
    var0 = df[df['T'] == 0]['noise'].var()
    Lambdas= np.arange(start=-4, stop=4 + 0.05, step=0.05)
    ATEs = []
    for Lambda in Lambdas:
        r1= Lambda * var1
        b1c = df['ps_0'] * r1
        b1 = b1c.mean()
        r0 = Lambda * var0
        b0c = df['ps_1'] * r0.mean()
        b0 = b0c.mean()
        fixed_ate = T_C - b1 - b0
        ATEs.append(fixed_ate)
    ate = min(np.abs(ATEs))
    if ate in ATEs :
        id1= ATEs .index(ate)
    else:
        id1 = ATEs.index(ate * (-1))

    TP = Lambdas[id1]
    plt.plot(Lambdas,ATEs)
    plt.ylim(-1.25,1.25)
    plt.xlim(-(4+0.05),4+0.05)
    plt.xlabel('Lambda')
    plt.ylabel('Fixed ATE')
    plt.axhline(y=0,linestyle=':')
    plt.savefig("threshold" + str(th) + ".png")
    plt.close()
    r1f = TP*var1
    r0f = TP*var0

    return r0f, r1f

#########################main################################################
def main():

    path = "Final_Data.csv"
    data, T, Y = read_data(path)
    X = data.iloc[:, :9]
    T_values = data.iloc[:, 9]
    file = open("scores.txt", "w")
    for Threshold in range(7, 21):
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
        matching=Matching(adj_data)
        file.write("Matching = %0.8f\n" % matching)
        adj_data2=adj_data.copy()
        doubly_robust_ate = doubly_robust(adj_data)
        file.write("Doubly Robust = %0.8f\n" % doubly_robust_ate)
        T_C, y_pred = log_liklihood(adj_data2)
        adj_data2['pred_y'] = y_pred
        sensitivity_analysis(adj_data2, T_C,Threshold)


    file.close()


if __name__ == "__main__":
    main()
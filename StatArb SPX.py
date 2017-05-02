'''
### StatArb SPX ###

Daily [long best k] & [short worst k] of top-market-cap candidates that were in the SPX at least once
Application of paper "DNNs, GBTs, RAFs: StatArb on the S&P500" (Krauss et al., 2016) to filter candidates by market cap
'''

import matplotlib.pyplot as plt
from Settings import *
from LoadData import *
from ML import *

'''
### Naming convention ###
constant   : xyz
list, range: xyzs (plural)
1D np.array: na_xyzs (plural); exception: y_train, y_test
>= 2D np.array: X_yz
data row number: m_xyz
'''

def Trade(Best_k, Worst_k, m_start, m_t):
    R_L = np.zeros((m_t, k))    # Rates of return of [long best k]
    R_S = np.zeros((m_t, k))    # Rates of return of [short worst k]

    for i in range(m_t):
        R_L[i] =   B[m_start+i+na_lookforwards,  Best_k[i]] / B[m_start+i,  Best_k[i]] - 1.0
        R_S[i] = - B[m_start+i+na_lookforwards, Worst_k[i]] / B[m_start+i, Worst_k[i]] + 1.0

    # Mean rates of return
    na_R_L_mean = R_L.mean(axis=1)
    na_R_S_mean = R_S.mean(axis=1)
    na_R_mean = na_R_L_mean + na_R_S_mean - trading_cost

    na_R_cumul = np.zeros(m_t)
    na_R_cumul[0] = na_R_mean[0]
    for i in range(1, m_t):
        if bool_compounded:
            na_R_cumul[i] = (1.0 + na_R_cumul[i-1]) * (1.0 + na_R_mean[i]) - 1.0
        else:
            na_R_cumul[i] = na_R_cumul[i-1] + na_R_mean[i]

    return na_R_mean, na_R_cumul


###########################################################
# The "GREAT for loop" begins !!!
###########################################################

period_Rs = []
print('\n# Number of batches: %d' % ((df_TRI.shape[0] - m_lb - m_train - m_lf) // m_test))
m_a = m_lb + m_train + m_test + m_lf
count = 0
# for m_b in range(0, df_TRI.shape[0] - m_a + 1, m_test):
for m_b in range(0, 1241+250 - m_a + 1, m_test):    # temporary setup for speed

    ### 3. Split dataset ###
    b_rows = range(m_b, m_b + m_a)
    print('\n############################################################')
    print(  '# Batch %2d rows:' % count, b_rows)  # select batch dataset

    m_c2 = m_lb + m_train
    candidates = []
    for j in Argsort_N[m_b]:  # only check those on the 1st row of batch
        if (Argsort_N[range(m_b, m_b + m_c2)] == j).sum() == m_c2:  # check each column for the period of m_lb + m_train
            candidates.append(j)
    n = len(candidates);    print('# n = %d' % n)
    print('############################################################')

    # Use "values" for speed in ML
    B = df_TRI.iloc[b_rows, candidates].values

    ### 4. Feature engineering ###

    ### Prepare X ###
    X_train_raw = np.zeros((m_train, n, p))
    X_test_raw = np.zeros((m_test, n, p))
    # R_X = np.zeros((p, n))

    m_c1 = m_lb
    for i in range(m_train):
        R_X = B[m_c1 + i] / B[m_c1 + i - na_lookbacks] - 1.0  # rate of return for each lookback
        X_train_raw[i] = R_X.T

    m_c2 = m_lb + m_train
    for i in range(m_test):
        R_X = B[m_c2 + i] / B[m_c2 + i - na_lookbacks] - 1.0  # rate of return for each lookback
        X_test_raw[i] = R_X.T

    X_train_r = X_train_raw.reshape((m_train * n, p))
    X_test_r = X_test_raw.reshape((m_test * n, p))

    ### Prepare y ###
    Y_train_raw = np.zeros((m_train, n))
    Y_test_raw = np.zeros((m_test, n))

    m_c1 = m_lb
    for i in range(m_train):
        na_R_y = B[m_c1 + i + na_lookforwards] / B[m_c1 + i] - 1.0  # rate of return for each lookforward
        Y_train_raw[i] = (na_R_y > np.median(na_R_y))  # whether outperformed (the cross-sectional median)

    m_c2 = m_lb + m_train
    for i in range(m_test):
        na_R_y = B[m_c2 + i + na_lookforwards] / B[m_c2 + i] - 1.0  # rate of return for each lookforward
        Y_test_raw[i] = (na_R_y > np.median(na_R_y))  # whether outperformed (the cross-sectional median)

    y_train_r = Y_train_raw.ravel()
    y_test_r = Y_test_raw.ravel()

    ### 4a. Feature scaling ###
    # Feature scaling by na_lookbacks (p)
    ssp = preprocessing.StandardScaler()
    X_train_ssp_r = ssp.fit_transform(X_train_r)
    X_test_ssp_r = ssp.transform(X_test_r)

    # Feature scaling by each day (m_train, m_test)
    X_train_ssm = np.zeros((m_train, n, p))
    X_test_ssm = np.zeros((m_test, n, p))
    ssm = preprocessing.StandardScaler()
    for i in range(m_train):
        X_train_ssm[i, :, :] = ssm.fit_transform(X_train_raw[i, :, :])
    for i in range(m_test):
        X_test_ssm[i, :, :] = ssm.fit_transform(X_test_raw[i, :, :])  # fit_transform for X_test as well
    X_train_ssm_r = X_train_ssm.reshape((m_train*n, p))
    X_test_ssm_r = X_test_ssm.reshape((m_test*n, p))

    ### 5. ML algos / Ensemble ###

    # DNN #
    if 'MLPClassifier' in algos:
        print('\n# MLPClassifier ############################################')
        Best_k_train_DNN, Worst_k_train_DNN, Best_k_test_DNN, Worst_k_test_DNN, model_DNN\
            = runMLPClassifier(X_train_ssm_r, y_train_r, X_test_ssm_r, y_test_r)

    # GBT #
    if 'XGBoost' in algos:
        print('\n# XGBoost ##################################################')
        Best_k_train_GBT, Worst_k_train_GBT, Best_k_test_GBT, Worst_k_test_GBT, model_GBT\
            = runXGBoost(X_train_r, y_train_r, X_test_r, y_test_r, n_round=20)
    if 'GradientBoostingClassifier' in algos:
        print('\n# GradientBoostingClassifier ###############################')
        Best_k_train_GBT, Worst_k_train_GBT, Best_k_test_GBT, Worst_k_test_GBT, model_GBT\
            = runGradientBoostingClassifier(X_train_r, y_train_r, X_test_r, y_test_r)
    if 'AdaBoostClassifier' in algos:
        print('\n# AdaBoostClassifier #######################################')
        Best_k_train_GBT, Worst_k_train_GBT, Best_k_test_GBT, Worst_k_test_GBT, model_GBT\
            = runAdaBoostClassifier(X_train_r, y_train_r, X_test_r, y_test_r)

    # RAF #
    if 'RandomForestClassifier' in algos:
        print('\n# RandomForestClassifier ###################################')
        Best_k_train_RAF, Worst_k_train_RAF, Best_k_test_RAF, Worst_k_test_RAF, model_RAF\
            = runRandomForestClassifier(X_train_r, y_train_r, X_test_r, y_test_r)

    # ENSEMBLE #
    if 'simple average' in ensembles:
        print('\n### simple average ENSEMBLE ################################')
        Best_k_train_ENS, Worst_k_train_ENS, Best_k_test_ENS, Worst_k_test_ENS = runENS_sa(y_train_r, y_test_r)


    ### 6. Performance metric ###

    # Trade training dataset
    na_R_mean_train_DNN, na_R_cumul_train_DNN = Trade(Best_k_train_DNN, Worst_k_train_DNN, m_lb, m_train)
    na_R_mean_train_GBT, na_R_cumul_train_GBT = Trade(Best_k_train_GBT, Worst_k_train_GBT, m_lb, m_train)
    na_R_mean_train_RAF, na_R_cumul_train_RAF = Trade(Best_k_train_RAF, Worst_k_train_RAF, m_lb, m_train)
    na_R_mean_train_ENS, na_R_cumul_train_ENS = Trade(Best_k_train_ENS, Worst_k_train_ENS, m_lb, m_train)

    # Trade testing dataset
    na_R_mean_test_DNN, na_R_cumul_test_DNN = Trade(Best_k_test_DNN, Worst_k_test_DNN, m_lb + m_train, m_test)
    na_R_mean_test_GBT, na_R_cumul_test_GBT = Trade(Best_k_test_GBT, Worst_k_test_GBT, m_lb + m_train, m_test)
    na_R_mean_test_RAF, na_R_cumul_test_RAF = Trade(Best_k_test_RAF, Worst_k_test_RAF, m_lb + m_train, m_test)
    na_R_mean_test_ENS, na_R_cumul_test_ENS = Trade(Best_k_test_ENS, Worst_k_test_ENS, m_lb + m_train, m_test)

    print('\nPeriod return of DNN:', na_R_cumul_test_DNN[-1])
    print(  'Period return of GBT:', na_R_cumul_test_GBT[-1])
    print(  'Period return of RAF:', na_R_cumul_test_RAF[-1])
    print(  'Period return of ENS:', na_R_cumul_test_ENS[-1])
    period_Rs.append([na_R_cumul_test_DNN[-1], na_R_cumul_test_GBT[-1], na_R_cumul_test_RAF[-1], na_R_cumul_test_ENS[-1]])
    count += 1

# End of the "GREAT for loop" !!!
Period_Rs = np.array(period_Rs)
if not bool_compounded:
    print('\nTOTAL return of DNN:', Period_Rs.sum(axis=0)[0])
    print(  'TOTAL return of GBT:', Period_Rs.sum(axis=0)[1])
    print(  'TOTAL return of RAF:', Period_Rs.sum(axis=0)[2])
    print(  'TOTAL return of ENS:', Period_Rs.sum(axis=0)[3])

print('\n\n############################################################')
print('Remnant rows:', range(m_b + m_test, df_TRI.shape[0]))

'''
    plt.plot(range(m_train), na_R_mean_train_DNN)
    plt.plot(range(m_train), na_R_mean_train_GBT)
    plt.plot(range(m_train), na_R_mean_train_RAF)
    plt.plot(range(m_train), na_R_mean_train_ENS)
    plt.legend(('DNN', 'GBT', 'RAF', 'ENS'))
    plt.show()

    plt.plot(range(m_train), na_R_cumul_train_DNN)
    plt.plot(range(m_train), na_R_cumul_train_GBT)
    plt.plot(range(m_train), na_R_cumul_train_RAF)
    plt.plot(range(m_train), na_R_cumul_train_ENS)
    plt.legend(('DNN', 'GBT', 'RAF', 'ENS'))
    plt.show()
        
    plt.plot(range(m_test), na_R_mean_test_DNN)
    plt.plot(range(m_test), na_R_mean_test_GBT)
    plt.plot(range(m_test), na_R_mean_test_RAF)
    plt.plot(range(m_test), na_R_mean_test_ENS)
    plt.legend(('DNN', 'GBT', 'RAF', 'ENS'))
    plt.show()

    plt.plot(range(m_test), na_R_cumul_test_DNN)
    plt.plot(range(m_test), na_R_cumul_test_GBT)
    plt.plot(range(m_test), na_R_cumul_test_RAF)
    plt.plot(range(m_test), na_R_cumul_test_ENS)
    plt.legend(('DNN', 'GBT', 'RAF', 'ENS'))
    plt.show()
'''

'''
### 7. MODEL ###

### 8. REAL data ###

### 9. Reap ###
# A prototype for future development
'''
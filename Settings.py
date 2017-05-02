import numpy as np

N = 500    # How many top market cap stocks to consider
k =  10    # How many to long & short

# How many to look back
na_lookbacks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
p = len(na_lookbacks)
m_lb = max(na_lookbacks)

m_train = 750    # num of rows of training dataset
m_test = 250    # num of rows of testing dataset

# How many to look forward
na_lookforwards = np.array([1])
q = len(na_lookforwards)
m_lf = max(na_lookforwards)

# ML algos to use
algos = [
    ### DNN ###
    'MLPClassifier',    # Multi-Layer Perceptron

    ### GBT ###
    'XGBoost',
#   'GradientBoostingClassifier',
#   'AdaBoostClassifier',

    ### RAF ###
    'RandomForestClassifier'
]

# Ensembles to use
ensembles = [
    'simple average',
#   'stacking'    # not implemented yet
]

trading_cost = (0.0) / 100.0    # (two-way) transaction cost:
                                # commissions (0.15~0.18%)*2 + tax on selling 0.002% + slippage ?%

bool_compounded = False    # compounded rate of return or not
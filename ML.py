# ML algos
from sklearn import model_selection, preprocessing, metrics
import numpy as np
from Settings import algos, m_train, m_test, k

# Accumulate probabilities for ENSEMBLE
prbbs_train = []
prbbs_test = []


# DNN ######################################################
def runMLPClassifier(X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(31, 10, 5),
                          activation='relu',  # 'identity', 'logistic', 'tanh'
                          solver='adam',  # 'lbfgs', 'sgd'
                          alpha=0.0005,  # regularization
                          learning_rate='adaptive',  # 'constant', 'invscaling'
                          learning_rate_init=0.05,  # only when solver='sgd' or 'adam'
                          beta_1=0.9, beta_2=0.999, epsilon=1e-8,  # only when solver='adam'
                          max_iter=400,
                          warm_start=True,
                          verbose=1,
                          random_state=1)

    params = [{'hidden_layer_sizes': [(31, 10, 5), (31, 15, 5)],
               'alpha': [0.01, 0.05], 'learning_rate_init': [0.01, 0.05]}]
#    runGridSearchCV(model, X_train, y_train, X_test, y_test, params, ['accuracy'])

    model.fit(X_train, y_train)
    eval_model(model, X_train, y_train, X_test, y_test)

    Prbbs_train_DNN = (model.predict_proba(X_train)[:, 1]).reshape((m_train, int(X_train.shape[0]/m_train)))  # [:,1] (probability to outperform)
    Prbbs_test_DNN = (model.predict_proba(X_test)[:, 1]).reshape((m_test, int(X_test.shape[0]/m_test)))
    prbbs_train.append(Prbbs_train_DNN)
    prbbs_test.append(Prbbs_test_DNN)

    Best_k_train_DNN, Worst_k_train_DNN = find_k(Prbbs_train_DNN)
    Best_k_test_DNN, Worst_k_test_DNN = find_k(Prbbs_test_DNN)
    return Best_k_train_DNN, Worst_k_train_DNN, Best_k_test_DNN, Worst_k_test_DNN, model


# GBT #####################################################
# Gradient Boosted Tree: combination of weak learners(shallow trees) into one strong learner
def runXGBoost(X_train, y_train, X_test, y_test, n_round=1000, seed_val=27):
    import xgboost as xgb
    dict_params = {}
    dict_params['objective'] = 'binary:logistic'
    dict_params['eval_metric'] = 'logloss'
    dict_params['eta'] = 0.3
    dict_params['max_depth'] = 6
    dict_params['min_child_weight'] = 1
    dict_params['subsample'] = 0.7
    dict_params['colsample_bytree'] = 0.7
    dict_params['silent'] = 1
    dict_params['seed'] = seed_val

    params = list(dict_params.items())
    Xg_train_set = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, Xg_train_set, n_round)

    Xg_train = xgb.DMatrix(X_train)
    Xg_test = xgb.DMatrix(X_test)
    Prbbs_train_GBT = (model.predict(Xg_train)).reshape(m_train, int(X_train.shape[0]/m_train))
    Prbbs_test_GBT = (model.predict(Xg_test)).reshape(m_test, int(X_train.shape[0]/m_train))
    prbbs_train.append(Prbbs_train_GBT)
    prbbs_test.append(Prbbs_test_GBT)

    na_preds_train_GBT = np.where(Prbbs_train_GBT > 0.5, 1., 0.).ravel()
    na_preds_test_GBT = np.where(Prbbs_test_GBT > 0.5, 1., 0.).ravel()

    print('training accuracy:', (na_preds_train_GBT == y_train).mean())
    print('testing  accuracy:', (na_preds_test_GBT == y_test).mean())
    print('training f1 score:', metrics.f1_score(y_train, na_preds_train_GBT))
    print('testing  f1 score:', metrics.f1_score(y_test, na_preds_test_GBT))

    Best_k_train_GBT, Worst_k_train_GBT = find_k(Prbbs_train_GBT)
    Best_k_test_GBT, Worst_k_test_GBT = find_k(Prbbs_test_GBT)
    return Best_k_train_GBT, Worst_k_train_GBT, Best_k_test_GBT, Worst_k_test_GBT, model

def runGradientBoostingClassifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, max_features=15,
                                       warm_start=True,
                                       verbose=1,
                                       random_state=1)
    # temporary setup for speed
    model = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=3, max_features=15, verbose=1, random_state=1)

    params = [{'n_estimators': [50, 100, 300], 'max_features': [5, 10, 15, 31]}]
    # runGridSearchCV(model, params, ['accuracy'])

    model.fit(X_train, y_train)
    eval_model(model, X_train, y_train, X_test, y_test)

    Prbbs_train_GBT = (model.predict_proba(X_train)[:, 1]).reshape((m_train, int(X_train.shape[0]/m_train)))  # [:,1] (probability to outperform)
    Prbbs_test_GBT = (model.predict_proba(X_test)[:, 1]).reshape((m_test, int(X_test.shape[0]/m_test)))
    prbbs_train.append(Prbbs_train_GBT)
    prbbs_test.append(Prbbs_test_GBT)

    Best_k_train_GBT, Worst_k_train_GBT = find_k(Prbbs_train_GBT)
    Best_k_test_GBT, Worst_k_test_GBT = find_k(Prbbs_test_GBT)
    return Best_k_train_GBT, Worst_k_train_GBT, Best_k_test_GBT, Worst_k_test_GBT, model

def runAdaBoostClassifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                               random_state=1)

    params = [{'n_estimators': [50, 100, 300]}]
    # runGridSearchCV(model, params, ['accuracy'])

    model.fit(X_train, y_train)
    eval_model(model, X_train, y_train, X_test, y_test)

    Prbbs_train_GBT = (model.predict_proba(X_train)[:, 1]).reshape((m_train, int(X_train.shape[0]/m_train)))  # [:,1] (probability to outperform)
    Prbbs_test_GBT = (model.predict_proba(X_test)[:, 1]).reshape((m_test, int(X_test.shape[0]/m_test)))
    prbbs_train.append(Prbbs_train_GBT)
    prbbs_test.append(Prbbs_test_GBT)

    Best_k_train_GBT, Worst_k_train_GBT = find_k(Prbbs_train_GBT)
    Best_k_test_GBT, Worst_k_test_GBT = find_k(Prbbs_test_GBT)
    return Best_k_train_GBT, Worst_k_train_GBT, Best_k_test_GBT, Worst_k_test_GBT, model


# RAF #####################################################
# Random Forest: Combination of de-correlated trees that use a random subset of training data(Bagging) and features --> majority vote
def runRandomForestClassifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                   max_depth=20,
                                   max_features='sqrt',    # 5
                                   # min_samples_split=5, min_samples_leaf=3,
                                   warm_start=True,
                                   n_jobs=-1,
                                   verbose=0,
                                   random_state=1)
    # temporary setup for speed
    model = RandomForestClassifier(n_estimators=10, max_depth=20, max_features=5, n_jobs=4, verbose=2, random_state=1)

    params = [{'n_estimators': [50, 100, 300], 'max_depth': [7, 10, 15]}]
    # runGridSearchCV(model, params, ['accuracy'])

    model.fit(X_train, y_train)
    model.verbose = 0  # be quiet now
    eval_model(model, X_train, y_train, X_test, y_test)

    Prbbs_train_RAF = (model.predict_proba(X_train)[:, 1]).reshape((m_train, int(X_train.shape[0]/m_train)))  # [:,1] (probability to outperform)
    Prbbs_test_RAF = (model.predict_proba(X_test)[:, 1]).reshape((m_test, int(X_test.shape[0]/m_test)))
    prbbs_train.append(Prbbs_train_RAF)
    prbbs_test.append(Prbbs_test_RAF)

    Best_k_train_RAF, Worst_k_train_RAF = find_k(Prbbs_train_RAF)
    Best_k_test_RAF, Worst_k_test_RAF = find_k(Prbbs_test_RAF)
    return Best_k_train_RAF, Worst_k_train_RAF, Best_k_test_RAF, Worst_k_test_RAF, model


# ENSEMBLE ################################################
def runENS_sa(y_train, y_test):
    Prbbs_train_ENS = sum(prbbs_train) / len(algos)
    Prbbs_test_ENS = sum(prbbs_test) / len(algos)
    prbbs_train.clear()    # clear for next batch
    prbbs_test.clear()

    na_preds_train_ENS = np.where(Prbbs_train_ENS > 0.5, 1., 0.).ravel()
    na_preds_test_ENS = np.where(Prbbs_test_ENS > 0.5, 1., 0.).ravel()

    print('training accuracy:', (na_preds_train_ENS == y_train).mean())
    print('testing  accuracy:', (na_preds_test_ENS == y_test).mean())
    print('training f1 score:', metrics.f1_score(y_train, na_preds_train_ENS))
    print('testing  f1 score:', metrics.f1_score(y_test, na_preds_test_ENS))

    Best_k_train_ENS, Worst_k_train_ENS = find_k(Prbbs_train_ENS)
    Best_k_test_ENS, Worst_k_test_ENS = find_k(Prbbs_test_ENS)
    return Best_k_train_ENS, Worst_k_train_ENS, Best_k_test_ENS, Worst_k_test_ENS


# Sub-functions of ML #####################################

# GridSearchCV : Exhaustive search of hyper-parameters for an estimator
def runGridSearchCV(model, X_train, y_train, X_test, y_test,
                    params, scorings=['accuracy']):    # others: 'f1', 'precision', 'recall', ...
    for scoring in scorings:
        print('\n# Tuning hyper-parameters for %s' % scoring)
        gcv = model_selection.GridSearchCV(model, params, scoring,
                                           cv=5,    # cross-validation splitting strategy
                                           n_jobs=-1,
                                           verbose=1)
        gcv.fit(X_train, y_train)

        means = gcv.cv_results_['mean_test_score']
        stds  = gcv.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gcv.cv_results_['params']):
            print('%.3f (+/-%.3f) for %r' % (mean, std * 2, params))
        print('\n# Best parameters on development set:', gcv.best_params_)

        print('\n# Scores computed on evaluation set:\n')
        print(metrics.classification_report(y_test, gcv.predict(X_test), digits=3))

    print(gcv)
    #print(gcv.cv_results_)

def eval_model(model, X_train, y_train, X_test, y_test):
    print('\ntraining accuracy:', model.score(X_train, y_train))
    print(  'testing  accuracy:', model.score(X_test, y_test))

    na_preds_train = model.predict(X_train)
    na_preds_test = model.predict(X_test)
    print('training f1 score:', metrics.f1_score(y_train, na_preds_train))
    print('testing  f1 score:', metrics.f1_score(y_test, na_preds_test))

# Sort and find [best k] & [worst k]
def find_k(Prbbs):
    Argsort = np.argsort(Prbbs)
    Best_k  = Argsort[:, -k:]
    Worst_k = Argsort[:, :k]
    return Best_k, Worst_k
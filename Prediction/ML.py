from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_array
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from gplearn_variations.gp_Standard import genetic
from gplearn_variations.gp_Standard import fitness
from gplearn_variations.gp_MaxDepth import genetic_withMaxDepth
from gplearn_variations.gp_MaxDepth import fitness as fitness_withMaxDepth

import warnings
warnings.filterwarnings("ignore")

import simplificationDTs

###################################
#####      GP Parameters      #####
###################################

population_size_value=500
generations_value=50
tournament_size_value=20
stopping_criteria_value=0.0
const_range_value=(-1, 1)
init_depth_value=(2, 6)
init_method_value='half and half'
function_set_value= ['add', 'sub', 'mul', 'max', 'min', 'div']
metric_value='rmse'
parsimony_coefficient_value=0.00001
p_crossover_value=0.9
p_subtree_mutation_value=0.01
p_hoist_mutation_value=0.01
p_point_mutation_value=0.01
p_point_replace_value=0.05
max_samples_value=1.0
warm_start_value=False
n_jobs_value=1
verbose_value=1
random_state_value=None

###################################
####      M3GP Parameters      ####
###################################

operators = [("+",2),("-",2),("*",2),("/",2)]
elitism_size = 1


def predictions(predicted_labels, list_labels):
    waf = metrics.f1_score(list_labels, predicted_labels, average='weighted')
    fmeasure_noninteract, fmeasure_interact = metrics.f1_score(list_labels, predicted_labels, average=None)
    precision = metrics.precision_score(list_labels, predicted_labels)
    recall = metrics.recall_score(list_labels, predicted_labels)
    accuracy = metrics.accuracy_score(list_labels, predicted_labels)
    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy


def writePredictions(predictions, y, path_output):
    """
    Write the predictions.
    :param predictions: list of predicted values;
    :param y: list of expected values;
    :param path_output: path of predictions file;
    :return: file with predictions.
    """
    file_predictions = open(path_output, 'w')
    file_predictions.write('Predicted_output' + '\t' + 'Expected_Output' + '\n')
    for i in range(len(y)):
        file_predictions.write(str(predictions[i]) + '\t' + str(y[i]) + '\n')
    file_predictions.close()


def performance_XGB(X_train, X_test, y_train, y_test, path_output_predictions, path_ouput_importance, feature_names):
    """
    Applies Random Forest Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, n_aspects);
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, n_aspects);
    :param y_train: the target values (proxy values) of the training set. The shape of the list is (n_samplesTrain);
    :param y_test: the target values (proxy values) of the test set. The shape of the list is (n_samplesTest);
    :param path_output_predictions: path of predictions file;
    :return: a predictions file and a correlation value on the test set.
    """
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(np.array(X_train), np.array(y_train))

    predictions_test = xgb_model.predict(np.array(X_test))
    predictions_train = xgb_model.predict(np.array(X_train))
    writePredictions(predictions_train.tolist(), y_train, path_output_predictions + '_TrainSet')
    writePredictions(predictions_test.tolist(), y_test, path_output_predictions + '_TestSet')

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_test, y_test)

    importances = xgb_model.feature_importances_
    xgb_importances = pd.Series(importances, index=feature_names)
    with open(path_ouput_importance + '.txt', "w") as txt_file:
        txt_file.write(str(xgb_importances))

    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy



def performance_RF(X_train, X_test, y_train, y_test, path_output_predictions, path_ouput_importance, feature_names):
    """
    Applies Random Forest Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, n_aspects);
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, n_aspects);
    :param y_train: the target values (proxy values) of the training set. The shape of the list is (n_samplesTrain);
    :param y_test: the target values (proxy values) of the test set. The shape of the list is (n_samplesTest);
    :param path_output_predictions: path of predictions file;
    :return: a predictions file and a correlation value on the test set.
    """
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    predictions_test = rf_model.predict(X_test)
    predictions_train = rf_model.predict(X_train)
    writePredictions(predictions_train, y_train, path_output_predictions + '_TrainSet')
    writePredictions(predictions_test, y_test, path_output_predictions + '_TestSet')

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_test, y_test)

    importances = rf_model.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    with open(path_ouput_importance + '.txt', "w") as txt_file:
        txt_file.write(str(forest_importances))

    fig, ax = pl.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(path_ouput_importance + ".png", bbox_inches="tight")
    pl.close()

    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        inner_tree.feature[index] = TREE_UNDEFINED
        ##print("Pruned {}".format(index))

def prune_duplicate_leaves(mdl):
    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)


def performance_DT(X_train, X_test, y_train, y_test, path_output_predictions, filename_Modeloutput, aspects,
                             d=None):
    """
    Applies Decision Tree Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, n_aspects);
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, n_aspects);
    :param y_train: the target values (proxy values) of the training set. The shape of the list is (n_samplesTrain);
    :param y_test: the target values (proxy values) of the test set. The shape of the list is (n_samplesTest);
    :param path_output_predictions: path of predictions file;
    :param filename_Modeloutput: path of model file;
    :param aspects: list of semantic aspects;
    :param d: maximum depth of trees (if None there is no maximum depth);
    :return: a predictions file, a model file, and a correlation value on the test set.
    """
    if d == None:
        clf = DecisionTreeClassifier()
    else:
        clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)

    prune_duplicate_leaves(clf)

    predictions_test = clf.predict(X_test)
    predictions_train = clf.predict(X_train)
    writePredictions(predictions_train, y_train, path_output_predictions + '_TrainSet')
    writePredictions(predictions_test, y_test, path_output_predictions + '_TestSet')

    importances = clf.feature_importances_
    file_importances = open(filename_Modeloutput + "_FeatureImportances", 'w')
    i = 0
    for importance in importances:
        file_importances.write(aspects[i] + '\t' + str(importance) + '\n')
        i = i + 1
    file_importances.close()

    file_depth = open(filename_Modeloutput + "_Depth", 'w')
    depth = clf.get_depth()
    file_depth.write('Depth' + '\t' + str(depth))
    file_depth.close()

    model = export_text(clf, aspects)
    with open(filename_Modeloutput, 'w') as file_Modeloutput:
        print(model, file=file_Modeloutput)

    file_rules = open(filename_Modeloutput + "_Simplification(corrected)", 'w')
    rules_list = simplificationDTs.tree_to_code(clf, aspects)
    for rule in rules_list:
        file_rules.write(rule + '\n')
    file_rules.close()

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_test, y_test)
    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy


def performance_GP(X_train, X_test, y_train, y_test, path_output_predictions, filename_Modeloutput):
    """
    Applies Genetic Programming Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, n_aspects);
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, n_aspects);
    :param y_train: the target values (proxy values) of the training set. The shape of the list is (n_samplesTrain);
    :param y_test: the target values (proxy values) of the test set. The shape of the list is (n_samplesTest);
    :param path_output_predictions: path of predictions file;
    :param filename_Modeloutput:  path of model file;
    :return:a predictions file, a model file, and a correlation value on the test set.
    """
    gp = genetic.SymbolicRegressor(population_size=population_size_value,
                           generations=generations_value,
                           tournament_size=tournament_size_value,
                           stopping_criteria=stopping_criteria_value,
                           const_range=const_range_value,
                           init_depth=init_depth_value,
                           init_method=init_method_value,
                           function_set=function_set_value,
                           metric=metric_value,
                           parsimony_coefficient=parsimony_coefficient_value,
                           p_crossover=p_crossover_value,
                           p_subtree_mutation=p_subtree_mutation_value,
                           p_hoist_mutation=p_hoist_mutation_value,
                           p_point_mutation=p_point_mutation_value,
                           p_point_replace=p_point_replace_value,
                           max_samples=max_samples_value,
                           warm_start=warm_start_value,
                           n_jobs=n_jobs_value,
                           verbose=verbose_value,
                           random_state=random_state_value)
    gp.fit(X_train, y_train)

    generation = 0
    for program in gp.best_individuals():
        generation = generation + 1

    X_train = check_array(X_train)
    _, gp.n_features = X_train.shape
    predictions_train = program.execute(X_train)
    writePredictions(predictions_train, y_train, path_output_predictions + '_TrainSet')

    X_test = check_array(X_test)
    _, gp.n_features = X_test.shape
    predictions_test = program.execute(X_test)
    writePredictions(predictions_test, y_test, path_output_predictions + '_TestSet')

    file_model = open(filename_Modeloutput, 'w')
    file_model.write(str(gp._program))
    file_model.close()

    predictions_labels_test = []
    for predicted_value in list(predictions_test):
        if predicted_value < 0.5:
            predicted_label = 0
        else:
            predicted_label = 1
        predictions_labels_test.append(predicted_label)
    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_labels_test, y_test)
    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy


def performance_GP_MaxDepth(X_train, X_test, y_train, y_test, path_output_predictions, filename_Modeloutput, d, operators=['add', 'sub', 'mul', 'max', 'min', 'div']):
    """
    Applies Genetic Programming Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, n_aspects);
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, n_aspects);
    :param y_train: the target values (proxy values) of the training set. The shape of the list is (n_samplesTrain);
    :param y_test: the target values (proxy values) of the test set. The shape of the list is (n_samplesTest);
    :param path_output_predictions: path of predictions file;
    :param filename_Modeloutput:  path of model file;
    :return:a predictions file, a model file, and a correlation value on the test set.
    """

    if d =="3":
        metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth3,
                                       greater_is_better=False)
    elif d=="4":
        metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth4,
                                           greater_is_better=False)
    elif d=="5":
        metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth5,
                                           greater_is_better=False)
    elif d=="6":
        metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth6,
                                           greater_is_better=False)
    elif d=="20":
        metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth20,
                                           greater_is_better=False)
    elif d=="6(incremental)":
        metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth6Incremental,
                                           greater_is_better=False)


    gp = genetic_withMaxDepth.SymbolicRegressor(population_size=population_size_value,
                           generations=generations_value,
                           tournament_size=tournament_size_value,
                           stopping_criteria=stopping_criteria_value,
                           const_range=const_range_value,
                           init_depth=init_depth_value,
                           init_method=init_method_value,
                           function_set=operators,
                           metric=metric_function,
                           parsimony_coefficient=parsimony_coefficient_value,
                           p_crossover=p_crossover_value,
                           p_subtree_mutation=p_subtree_mutation_value,
                           p_hoist_mutation=p_hoist_mutation_value,
                           p_point_mutation=p_point_mutation_value,
                           p_point_replace=p_point_replace_value,
                           max_samples=max_samples_value,
                           warm_start=warm_start_value,
                           n_jobs=n_jobs_value,
                           verbose=verbose_value,
                           random_state=random_state_value)
    gp.fit(X_train, y_train)

    generation = 0
    for program in gp.best_individuals():
        generation = generation + 1

    X_train = check_array(X_train)
    _, gp.n_features = X_train.shape
    predictions_train = program.execute(X_train)
    writePredictions(predictions_train, y_train, path_output_predictions + '_TrainSet')

    X_test = check_array(X_test)
    _, gp.n_features = X_test.shape
    predictions_test = program.execute(X_test)
    writePredictions(predictions_test, y_test, path_output_predictions + '_TestSet')

    file_model = open(filename_Modeloutput, 'w')
    file_model.write(str(gp._program))
    file_model.close()

    predictions_labels_test = []
    for predicted_value in list(predictions_test):
        if predicted_value < 0.5:
            predicted_label = 0
        else:
            predicted_label = 1
        predictions_labels_test.append(predicted_label)
    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_labels_test, y_test)
    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy


def _fitness_function_maxDepth6(y, y_pred, sample_weight, depth):
    rmse = np.sqrt(np.average((y_pred - y) ** 2))
    if depth > 6:
        rmse = rmse * 10
    return rmse

def _fitness_function_maxDepth3(y, y_pred, sample_weight, depth):
    rmse = np.sqrt(np.average((y_pred - y) ** 2))
    if depth > 3:
        rmse = rmse * 10
    return rmse


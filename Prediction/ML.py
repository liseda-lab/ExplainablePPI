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
import pickle

from gplearn_variations.gp_Standard import genetic
from gplearn_variations.gp_Standard import fitness
from gplearn_variations.gp_MaxDepth import genetic_withMaxDepth
from gplearn_variations.gp_MaxDepth import fitness as fitness_withMaxDepth

import warnings
warnings.filterwarnings("ignore")

from Prediction import processDTs
from Prediction import processGPs
from Prediction import LIMEexplanations

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
    with open(path_output, 'w') as file_predictions:
        file_predictions.write('Predicted_output' + '\t' + 'Expected_Output' + '\n')
        for i in range(len(y)):
            file_predictions.write(str(predictions[i]) + '\t' + str(y[i]) + '\n')


def writeEvaluationPredictions(waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, path_output_evaluation_prediction):
    with open(path_output_evaluation_prediction, 'w') as file_evaluation:
        file_evaluation.write("WAF\t" + str(waf) + '\n')
        file_evaluation.write("F-measure(class0)\t" + str(fmeasure_noninteract) + '\n')
        file_evaluation.write("F-measure(class1)\t" + str(fmeasure_interact) + '\n')
        file_evaluation.write("Precision\t" + str(precision) + '\n')
        file_evaluation.write("Recall\t" + str(recall) + '\n')
        file_evaluation.write("Accuracy\t" + str(accuracy) + '\n')


def performance_XGB(pairs_test, list_ss, list_labels, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_ouput_model, path_output_evaluation_prediction, path_output_explanation):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(np.array(X_train), np.array(y_train))

    predictions_test = xgb_model.predict(np.array(X_test))
    writePredictions(predictions_test.tolist(), y_test, path_output_predictions)

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_test, y_test)
    writeEvaluationPredictions(waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, path_output_evaluation_prediction)

    pickle.dump(xgb_model, open(path_ouput_model, 'wb'))

    with open(path_output_explanation, 'w') as path_output_explanation:
        path_output_explanation.write('Ent1\tEnt1\tLIME3\tLIME8\tLORE1\tLORE2\n')
        for i in range(len(y_test)):
            lime3_explanation = LIMEexplanations.LIME_explanation(X_train, X_test, i, xgb_model, aspects, 3)
            lime8_explanation = LIMEexplanations.LIME_explanation(X_train, X_test, i, xgb_model, aspects, 8)

            path_output_explanation.write(pairs_test[i][0] + '\t' + pairs_test[i][1] + '\t' + str(lime3_explanation) + '\t' + str(lime8_explanation) + '\n')


def performance_RF(pairs_test, list_ss, list_labels, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_ouput_model, path_output_evaluation_prediction, path_output_explanation):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    predictions_test = rf_model.predict(X_test)
    writePredictions(predictions_test, y_test, path_output_predictions)

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_test, y_test)
    writeEvaluationPredictions(waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, path_output_evaluation_prediction)

    pickle.dump(rf_model, open(path_ouput_model, 'wb'))

    with open(path_output_explanation, 'w') as path_output_explanation:
        path_output_explanation.write('Ent1\tEnt1\tLIME3\tLIME8\tLORE1\tLORE2\n')
        for i in range(len(y_test)):
            lime3_explanation = LIMEexplanations.LIME_explanation(X_train, X_test, i, rf_model, aspects, 3)
            lime8_explanation = LIMEexplanations.LIME_explanation(X_train, X_test, i, rf_model, aspects, 8)

            path_output_explanation.write(pairs_test[i][0] + '\t' + pairs_test[i][1] + '\t' + str(lime3_explanation) + '\t' + str(lime8_explanation) + '\n')


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


def performance_DT(dict_SS, pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_ouput_model, path_output_evaluation_prediction, path_output_explanation, d=None):
    if d == None:
        clf = DecisionTreeClassifier()
    else:
        clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)

    prune_duplicate_leaves(clf)
    predictions_test = clf.predict(X_test)
    writePredictions(predictions_test, y_test, path_output_predictions)

    model = export_text(clf, aspects)
    with open(path_ouput_model, 'w') as file_Modeloutput:
        print(model, file=file_Modeloutput)

    rules_list = processDTs.tree_to_code(clf, aspects)
    with open(path_output_explanation, 'w') as file_rules:
        for ent1, ent2 in pairs_test:
            rule = processDTs.get_decision_path(rules_list, aspects, dict_SS, ent1, ent2)
            file_rules.write(ent1 + '\t' + ent2 + '\t' + rule + '\n')

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_test, y_test)
    writeEvaluationPredictions(waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, path_output_evaluation_prediction)


def performance_GP(pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_ouput_model, path_output_evaluation_prediction, path_output_explanation):
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

    X_test = check_array(X_test)
    _, gp.n_features = X_test.shape
    predictions_test = program.execute(X_test)
    writePredictions(predictions_test, y_test, path_output_predictions)

    with open(path_ouput_model, 'w') as file_model:
        expression_simplified = processGPs.processGPmodel(str(gp._program), aspects)
        file_model.write(str(expression_simplified))

    predictions_labels_test = []
    for predicted_value in list(predictions_test):
        if predicted_value < 0.5:
            predicted_label = 0
        else:
            predicted_label = 1
        predictions_labels_test.append(predicted_label)
    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_labels_test, y_test)
    writeEvaluationPredictions(waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy,
                               path_output_evaluation_prediction)

    with open(path_output_explanation, 'w') as file_explanations:
        for ent1, ent2 in pairs_test:
            file_explanations.write(ent1 + '\t' + ent2 + '\t' + str(expression_simplified) + '\n')


def performance_GP6x(pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_ouput_model, path_output_evaluation_prediction, path_output_explanation):
    metric_function = fitness_withMaxDepth._Fitness(_fitness_function_maxDepth6, greater_is_better=False)
    gp = genetic_withMaxDepth.SymbolicRegressor(population_size=population_size_value,
                           generations=generations_value,
                           tournament_size=tournament_size_value,
                           stopping_criteria=stopping_criteria_value,
                           const_range=const_range_value,
                           init_depth=init_depth_value,
                           init_method=init_method_value,
                           function_set=['add', 'sub', 'max', 'min'],
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

    X_test = check_array(X_test)
    _, gp.n_features = X_test.shape
    predictions_test = program.execute(X_test)
    writePredictions(predictions_test, y_test, path_output_predictions)

    with open(path_ouput_model, 'w') as file_model:
        expression_simplified = processGPs.processGPmodel(str(gp._program), aspects)
        file_model.write(str(expression_simplified))

    predictions_labels_test = []
    for predicted_value in list(predictions_test):
        if predicted_value < 0.5:
            predicted_label = 0
        else:
            predicted_label = 1
        predictions_labels_test.append(predicted_label)
    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = predictions(predictions_labels_test, y_test)
    writeEvaluationPredictions(waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy,
                               path_output_evaluation_prediction)

    with open(path_output_explanation, 'w') as file_explanations:
        for ent1, ent2 in pairs_test:
            file_explanations.write(ent1 + '\t' + ent2 + '\t' + str(expression_simplified) + '\n')


def _fitness_function_maxDepth6(y, y_pred, sample_weight, depth):
    rmse = np.sqrt(np.average((y_pred - y) ** 2))
    if depth > 6:
        rmse = rmse * 10
    return rmse




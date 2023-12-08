import numpy as np
import lime
import lime.lime_tabular


def LIME_explanation(X_train,X_test,i,ml_model, aspects, features=6):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                                                            feature_names=aspects,
                                                            discretize_continuous=True)
    exp = lime_explainer.explain_instance(np.array(X_test[i]), ml_model.predict_proba, num_features=features)
    return str(exp.as_list())


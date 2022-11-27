import os
import uuid
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# This is used in order to show the plotted figures within this notebook
# %load_ext tensorboard
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from math import ceil
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_roc_curve, roc_curve, auc, make_scorer
from sklearn.model_selection import learning_curve
from sklearn.multioutput import MultiOutputClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

data = pd.read_csv('output/loan.csv')
data = data.sort_values(by='loan_date')
competition = data[data['Predicted'].isna()]
data = data[~data['Predicted'].isna()]

# These columns will be used as the inputs of the models
input_cols = ['Id', 'loan_date', 'loan_duration', 'loan_payments',
              'account_district_code', 'account_frequency', 'account_date',
              'account_district_name', 'account_district_region',
              'account_district_no_inhabitants',
              'account_district_no_municipalities_0_499',
              'account_district_no_municipalities_500_1999',
              'account_district_no_municipalities_2000_9999',
              'account_district_no_municipalities_10000_plus',
              'account_district_no_cities',
              'account_district_ratio_urban_inhabitants',
              'account_district_average_salary',
              # 'account_district_unemployment_rate_95',
              'account_district_unemployment_rate_96',
              'account_district_no_enterpreneurs_per_1000_inhabitants',
              # 'account_district_no_crimes_95',
              'account_district_no_crimes_96',
              'owner_district_code', 'owner_card_type',
              # 'owner_card_issued',
              'owner_district_name', 'owner_district_region',
              'owner_district_no_inhabitants',
              'owner_district_no_municipalities_0_499',
              'owner_district_no_municipalities_500_1999',
              'owner_district_no_municipalities_2000_9999',
              'owner_district_no_municipalities_10000_plus',
              'owner_district_no_cities', 'owner_district_ratio_urban_inhabitants',
              'owner_district_average_salary',
              # 'owner_district_unemployment_rate_95',
              'owner_district_unemployment_rate_96',
              'owner_district_no_enterpreneurs_per_1000_inhabitants',
              # 'owner_district_no_crimes_95',
              'owner_district_no_crimes_96',
              'disponent_district_code', 'disponent_district_name',
              'disponent_district_region',
              # 'disponent_district_no_inhabitants',
              # 'disponent_district_no_municipalities_0_499',
              # 'disponent_district_no_municipalities_500_1999',
              # 'disponent_district_no_municipalities_2000_9999',
              # 'disponent_district_no_municipalities_10000_plus',
              # 'disponent_district_no_cities',
              # 'disponent_district_ratio_urban_inhabitants',
              # 'disponent_district_average_salary',
              # 'disponent_district_unemployment_rate_95',
              # 'disponent_district_unemployment_rate_96',
              # 'disponent_district_no_enterpreneurs_per_1000_inhabitants',
              # 'disponent_district_no_crimes_95', 'disponent_district_no_crimes_96',
              'count_trans_credits', 'count_trans_withdrawals',
              'count_trans_credit_cash', 'count_trans_withdrawal_cash',
              'count_trans_withdrawal_card', 'count_trans_collection_other_bank',
              'count_trans_remittance_other_bank',
              'count_trans_ksymbol_interest_credited',
              'count_trans_ksymbol_household',
              'count_trans_ksymbol_payment_for_statement',
              'count_trans_ksymbol_insurance_payment',
              'count_trans_ksymbol_sanction_interest_if_negative_balance',
              'count_trans_ksymbol_oldage_pension', 'last_trans_balance',
              'mean_trans_balance', 'mean_trans_amount_absolute',
              'mean_trans_amount_credit',
              # 'mean_trans_amount_withdrawal',
              'mean_trans_amount_signed', 'owner_male', 'owner_birthdate',
              # 'disponent_male', 'disponent_birthdate'
              ]

# The output columns are the genres
output_cols = ['Predicted']

# Averages to calculate for precision, recall, and f1-score
averages = [None, "macro", "weighted", "micro", "samples"]

feat_enc = LabelEncoder()
data['account_frequency'] = feat_enc.fit_transform(data['account_frequency'])
data['owner_card_type'] = feat_enc.fit_transform(data['owner_card_type'])
data['account_district_code'] = feat_enc.fit_transform(
    data['account_district_code'])
data['account_district_name'] = feat_enc.fit_transform(
    data['account_district_name'])
data['account_district_region'] = feat_enc.fit_transform(
    data['account_district_region'])
data['owner_district_code'] = feat_enc.fit_transform(
    data['owner_district_code'])
data['owner_district_name'] = feat_enc.fit_transform(
    data['owner_district_name'])
data['owner_district_region'] = feat_enc.fit_transform(
    data['owner_district_region'])
data['disponent_district_code'] = feat_enc.fit_transform(
    data['disponent_district_code'])
data['disponent_district_name'] = feat_enc.fit_transform(
    data['disponent_district_name'])
data['disponent_district_region'] = feat_enc.fit_transform(
    data['disponent_district_region'])

competition['account_frequency'] = feat_enc.fit_transform(
    competition['account_frequency'])
competition['owner_card_type'] = feat_enc.fit_transform(
    competition['owner_card_type'])
competition['account_district_code'] = feat_enc.fit_transform(
    competition['account_district_code'])
competition['account_district_name'] = feat_enc.fit_transform(
    competition['account_district_name'])
competition['account_district_region'] = feat_enc.fit_transform(
    competition['account_district_region'])
competition['owner_district_code'] = feat_enc.fit_transform(
    competition['owner_district_code'])
competition['owner_district_name'] = feat_enc.fit_transform(
    competition['owner_district_name'])
competition['owner_district_region'] = feat_enc.fit_transform(
    competition['owner_district_region'])
competition['disponent_district_code'] = feat_enc.fit_transform(
    competition['disponent_district_code'])
competition['disponent_district_name'] = feat_enc.fit_transform(
    competition['disponent_district_name'])
competition['disponent_district_region'] = feat_enc.fit_transform(
    competition['disponent_district_region'])

data.replace('?', np.nan, inplace=True)
competition.replace('?', np.nan, inplace=True)

data.isnull().any()

# Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def plot_learning_curve(
    title,
    train_sizes,
    train_scores,
    test_scores,
    fit_times,
    score_times,
    axes=None,
    ylim=None,
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    axes = axes.reshape(-1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig = fig.delaxes(axes[-1])

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    # Plot n_samples vs score_times
    axes[3].grid()
    axes[3].plot(train_sizes, score_times_mean, "o-")
    axes[3].fill_between(
        train_sizes,
        score_times_mean - score_times_std,
        score_times_mean + score_times_std,
        alpha=0.1,
    )
    axes[3].set_xlabel("Training examples")
    axes[3].set_ylabel("score_times")
    axes[3].set_title("Scalability of the model")

    # Plot score_time vs score
    score_time_argsort = score_times_mean.argsort()
    score_time_sorted = score_times_mean[score_time_argsort]
    test_scores_mean_sorted = test_scores_mean[score_time_argsort]
    test_scores_std_sorted = test_scores_std[score_time_argsort]
    axes[4].grid()
    axes[4].plot(score_time_sorted, test_scores_mean_sorted, "o-")
    axes[4].fill_between(
        score_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[4].set_xlabel("score_times")
    axes[4].set_ylabel("Score")
    axes[4].set_title("Performance of the model")

    return plt


# The following helper functions are for training and evaluating the model

def show_confusion_matrix(cms, target_names, output_labels, title):
    """
    This helper function plots the confusion matrices calculated when evaluating the model.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.suptitle(title, fontsize=32)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    gnames = ["True Negative", "False Positive",
              "False Negative", "True Positive"]
    gcounts = [f"{v:0.0f}" for v in cms.flatten()]
    gpercentages = [f"{v:.2%}" for v in cms.flatten()/np.sum(cms)]
    annot = np.asarray([f"{name}\n{count}\n{percentage}" for name, count, percentage in zip(
        gnames, gcounts, gpercentages)]).reshape(2, 2)

    sns.heatmap(cms, ax=ax, annot=annot, fmt="", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')


def evaluate_model(model, testing_inputs, testing_classes, output_cols, sample_weight=None):
    """
    This helper function prints the report and evaluation metrics for the model.
    """
    predictions = model.predict(testing_inputs)

    print("="*70)
    print(f"Evaluation metrics for {model.__class__.__name__}")
    print("="*70)

    score = model.score(testing_inputs, testing_classes)
    print(f"{model.__class__.__name__}'s default score metric: {score}")

    print("Classification report")
    print(
        classification_report(testing_classes, predictions,
                              sample_weight=sample_weight, digits=4, zero_division=1)
    )

    accuracy = accuracy_score(
        testing_classes, predictions, sample_weight=sample_weight)
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"AUC: {roc_auc_score(testing_classes, predictions, sample_weight=sample_weight):.4f}")

    cms = confusion_matrix(testing_classes, predictions,
                           sample_weight=sample_weight)
    show_confusion_matrix(cms, ['no', 'yes'], output_cols,
                          f"Confusion matrices for {model.__class__.__name__}")

    print("="*70)


def train_and_evaluate(input_cols, output_cols, model, params, scoring, n_iter=None, sample_weight=None, random_state=42, plot_roc=True):
    """
    This function trains the model and prints the evaluation metrics, as well as the confusion matrices, and learning and scalability plots.
    """
    inputs = data[input_cols].values
    classes = data[output_cols].values
    (training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(
        inputs, classes, test_size=0.4, shuffle=False, random_state=random_state)

    # Define SMOTE-Tomek Links
    resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    training_inputs, training_classes = resample.fit_resample(
        training_inputs, training_classes)

    if n_iter == None:
        clf = GridSearchCV(model, params, n_jobs=1,
                           cv=TimeSeriesSplit(), scoring=scoring, verbose=2)
    else:
        clf = RandomizedSearchCV(
            model, params, n_iter=n_iter, scoring=scoring,
            n_jobs=1, cv=TimeSeriesSplit(), random_state=random_state, verbose=2)

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        clf, training_inputs, training_classes, scoring=scoring, return_times=True, cv=5, n_jobs=1, random_state=random_state)

    plot_learning_curve(f"Learning curves for {model.__class__.__name__}",
                        train_sizes, train_scores, test_scores, fit_times, score_times)

    resclf = clf.fit(training_inputs, training_classes)

    if plot_roc:
        plot_roc_curve(resclf, testing_inputs, testing_classes)

    if isinstance(model, DecisionTreeClassifier):
        plot_tree(resclf.best_estimator_, feature_names=input_cols)
        plt.savefig(f'output/{model.__class__.__name__}_tree_diagram.svg')

    print(f"Best params for {model.__class__.__name__}: {clf.best_params_}")

    evaluate_model(clf, testing_inputs, testing_classes,
                   output_cols, sample_weight=sample_weight)
    return clf


def use_model(model, params, scoring='f1_weighted', n_iter=None, sample_weight=None, random_state=42, plot_roc=True):
    """
    A more convenient wrapper around train_and_evaluate, albeit less general.
    """
    clf = train_and_evaluate(input_cols, output_cols, model, params, n_iter=n_iter,
                             sample_weight=sample_weight, random_state=random_state, plot_roc=plot_roc, scoring=scoring)
    inputs = competition[input_cols].values
    results = clf.predict(inputs)
    print(results)


dt = use_model(
    DecisionTreeClassifier(random_state=42),
    {
        "criterion": ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        "max_depth": range(1, 20),
        'max_features': range(1, 40),
        "min_samples_split": range(2, 15),
        "min_samples_leaf": range(1, 7)
    },
)

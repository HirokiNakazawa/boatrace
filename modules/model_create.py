"""
モデル作成関連の関数群
"""

from typing import Tuple
import pandas as pd
import optuna.integration.lightgbm as lgb_o
import lightgbm as lgb


def split_data(df: pd.DataFrame, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データを分割する
    """
    sorted_id_list = df.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(
        len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(
        len(sorted_id_list) * (1 - test_size)):]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train, test


def get_train_valid(results_c: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    訓練データと検証データを返す
    """
    train, test = split_data(results_c)
    train, valid = split_data(train)
    X_train = train.drop(["rank", "date"], axis=1)
    y_train = train["rank"]
    X_valid = valid.drop(["rank", "date"], axis=1)
    y_valid = valid["rank"]
    return X_train, y_train, X_valid, y_valid


def get_lgb_train_valid(results: pd.DataFrame) -> Tuple[lgb_o.Dataset, lgb_o.Dataset, dict]:
    """
    optuna用にデータを用意
    """
    X_train, y_train, X_valid, y_valid = get_train_valid(results)

    lgb_train = lgb_o.Dataset(X_train.values, y_train.values)
    lgb_valid = lgb_o.Dataset(X_valid.values, y_valid.values)

    params = {
        "objective": "binary",
        "random_state": 100,
        "metric": "binary_logloss",
        "verbosity": -1
    }
    return lgb_train, lgb_valid, params


def get_train_test(results_c: pd.DataFrame) -> pd.DataFrame:
    """
    訓練データとテストデータを返す
    """
    train, test = split_data(results_c)
    X_train = train.drop(["rank", "date"], axis=1)
    y_train = train["rank"]
    X_test = test.drop(["rank", "date"], axis=1)
    y_test = test["rank"]
    return X_train, y_train, X_test, y_test


def get_optuna_params(params: dict, lgb_train: lgb_o.Dataset, lgb_valid: lgb_o.Dataset, seed) -> dict:
    """
    パラメータチューニングを行なった結果を返す
    """
    verbose_eval = 0
    lgb_clf_o = lgb_o.train(
        params, lgb_train,
        valid_sets=[lgb_valid],
        optuna_seed=seed,
        callbacks=[lgb.early_stopping(
            stopping_rounds=10, verbose=True), lgb.log_evaluation(verbose_eval)]
    )
    params_o = lgb_clf_o.params
    del params_o["early_stopping_round"]
    return params_o


def get_lgb_clf(params_o: dict, X_train: pd.DataFrame, y_train: pd.DataFrame) -> lgb.LGBMClassifier:
    """
    モデルを作成し、返す
    """
    lgb_clf = lgb.LGBMClassifier(**params_o)
    lgb_clf.fit(X_train.values, y_train.values)
    return lgb_clf

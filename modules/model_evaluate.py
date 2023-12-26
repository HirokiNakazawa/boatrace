"""
モデル評価関連の関数群
"""

from typing import Tuple, Callable
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import lightgbm as lgb
import json
import pickle

from model import ModelEvaluator


def gain(return_func: Callable[[float], Tuple[int, float]], n_samples: int = 100) -> pd.DataFrame:
    """
    閾値ごとの回収率を算出する
    """
    gain = {}
    try:
        for i in tqdm(range(n_samples)):
            threshold = 0.5 + (((1 - 0.5) / n_samples) * i)
            df_hits, n_bets, use_money, return_rate = return_func(
                threshold)
            gain[threshold] = {"n_bets": n_bets,
                               "return_rate": return_rate,
                               "use_money": use_money}
        return pd.DataFrame(gain).T
    except:
        return pd.DataFrame(gain).T


def plot(df):
    plt.plot(df.index, df["return_rate"])


def get_gain_dict(lgb_clf: lgb.LGBMClassifier, returns: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """
    モデルを評価し、gainの辞書を返す
    """
    me = ModelEvaluator(lgb_clf, returns, X_test)
    gain_t = gain(me.tansho_return)
    gain_f = gain(me.fukusho_return)
    gain_2t = gain(me.nirentan_return)
    gain_2f = gain(me.nirenpuku_return)
    gain_3t = gain(me.sanrentan_return)
    gain_3f = gain(me.sanrenpuku_return)
    gain_3t_1 = gain(me.sanrentan_nagashi_1)
    gain_3t_2 = gain(me.sanrentan_nagashi_2)
    gain_3t_12b = gain(me.sanrentan_12_box)
    gain_3t_12b_n = gain(me.sanrentan_12_box_nagashi)
    gain_3t_b = gain(me.sanrentan_box)

    gain_dict = {"gain_t": gain_t, "gain_f": gain_f, "gain_2t": gain_2t, "gain_2f": gain_2f, "gain_3t": gain_3t, "gain_3f": gain_3f,
                 "gain_3t_1": gain_3t_1, "gain_3t_2": gain_3t_2, "gain_3t_12b": gain_3t_12b, "gain_3t_12b_n": gain_3t_12b_n, "gain_3t_b": gain_3t_b}
    return gain_dict


def change_format_gain_dict(gain_dict: dict) -> dict:
    """
    gainの辞書をJSON形式にフォーマットし、返す
    """
    gains = {}
    for k, v in gain_dict.items():
        datas = {}
        data = v[(v["n_bets"] > 100) & (v["return_rate"] > 100)]
        if not data.empty:
            for index, row in data.iterrows():
                datas[index] = {}
                datas[index]["n_bets"] = row["n_bets"]
                datas[index]["return_rate"] = row["return_rate"]
                datas[index]["return_money"] = (
                    row["use_money"] * row["return_rate"] * 0.01) - row["use_money"]
            gains[k] = datas
        else:
            pass
    return gains


def save_gain_dict(gain_dict: dict, seed: int) -> None:
    """
    gainの辞書を保存する
    """
    gains = change_format_gain_dict(gain_dict)
    file_name = f"gain/gain_{seed}.json"
    with open(file_name, mode="w") as f:
        json.dump(gains, f, ensure_ascii=False)


def save_model(model: lgb.LGBMClassifier, seed: int) -> None:
    """
    モデルを保存する
    """
    file_name = f"params/model_{seed}.pickle"
    with open(file_name, mode="wb") as f:
        pickle.dump(model, f)


def check_model(returns, X, seed):
    """
    モデルの回収率を確認する
    """
    file_name = f"params/model_{seed}.pickle"
    with open(file_name, mode="rb") as f:
        lgb_clf = pickle.load(f)

    gain_dict = get_gain_dict(lgb_clf, returns, X)
    gains = change_format_gain_dict(gain_dict)

    confirm_file_name = f"gain/confirm_gain_{seed}.json"
    with open(confirm_file_name, "w") as f:
        json.dump(gains, f, ensure_ascii=False)

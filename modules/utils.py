"""
共通関数群
競争成績URL :https://www1.mbrace.or.jp/od2/K/202201/01/02.html
番組表URL   :https://www1.mbrace.or.jp/od2/B/202201/01/02.html
"""

from pathlib import Path
import pandas as pd

from results import Results
from infos import Infos
from returns import Returns


def get_results_p(results: pd.DataFrame) -> pd.DataFrame:
    """
    整形したレース結果データを返す
    """
    r = Results(results)
    r.preprocessing()
    return r.results_p


def get_infos_p(infos: pd.DataFrame) -> pd.DataFrame:
    """
    整形したレース情報データを返す
    """
    i = Infos(infos)
    i.preprocessing()
    return i.infos_p


def get_returns_p(returns: pd.DataFrame) -> pd.DataFrame:
    """
    整形した払い戻し表データを返す
    """
    rt = Returns(returns)
    rt.preprocessing()
    return rt.returns_p


def get_results_merge_infos(results_p: pd.DataFrame, infos_p: pd.DataFrame) -> pd.DataFrame:
    """
    resultsとinfosを結合したデータを返す
    """
    results_all = pd.merge(results_p, infos_p, on=[
                           "race_id", "boat_number", "racer_number"], how="left")
    return results_all


def process_categorical(results: pd.DataFrame, is_predict: bool = False):
    df = results.copy()

    class_mapping = {"B2": 1, "B1": 2, "A2": 3, "A1": 4}
    df["class"] = df["class"].map(class_mapping)

    df["boat_number"] = df["boat_number"].astype("category")
    df["racer_number"] = df["racer_number"].astype("category")
    df["class"] = df["class"].astype("category")

    if is_predict:
        df.drop("date", axis=1, inplace=True)
    else:
        df["rank"] = df["position"].map(lambda x: 1 if x <= 2 else 0)
        df.drop("position", axis=1, inplace=True)
    return df

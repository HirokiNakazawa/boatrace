"""
予測関連の関数群
"""

import lightgbm as lgb
import pandas as pd
import pickle

from modules.utils import *


def predict_proba(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.Series:
    def standard_scaler(x):
        return (x - x.mean()) / x.std(ddof=0)

    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    proba = proba.groupby(level=0).transform(standard_scaler)
    proba = (proba - proba.min()) / (proba.max() - proba.min())
    return proba


def rank_join(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()[["boat_number"]]
    df["point"] = predict_proba(model, X)
    df["rank"] = df["point"].groupby(level=0).rank(ascending=False)
    return df


def pred_table(model: lgb.LGBMClassifier, target_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = rank_join(model, target_df)
    df["pred_0"] = [0 if p < threshold else 1 for p in df["point"]]
    df["pred_1"] = ((df["pred_0"] == 1) & (df["rank"] == 1)) * 1
    df["pred_2"] = ((df["pred_0"] == 1) & (df["rank"] == 2)) * 1
    df["pred_3"] = ((df["pred_0"] == 1) & (df["rank"] == 3)) * 1
    return df


def preprocessing_3(model: lgb.LGBMClassifier, target_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    上位3艇の予想データを返す
    """
    df = pred_table(model, target_df, threshold)
    df_3_1 = pd.DataFrame(df[df["pred_1"] == 1]["boat_number"]).rename(
        columns={"boat_number": "pred_1"})
    df_3_2 = pd.DataFrame(df[df["pred_2"] == 1]["boat_number"]).rename(
        columns={"boat_number": "pred_2"})
    df_3_3 = pd.DataFrame(df[df["pred_3"] == 1]["boat_number"]).rename(
        columns={"boat_number": "pred_3"})
    df_3_12 = pd.merge(df_3_1, df_3_2, left_index=True,
                       right_index=True, how="right")
    df_3 = pd.merge(df_3_12, df_3_3, left_index=True,
                    right_index=True, how="right")
    df_3["pred_1"] = df_3["pred_1"].astype(int)
    df_3["pred_2"] = df_3["pred_2"].astype(int)
    df_3["pred_3"] = df_3["pred_3"].astype(int)
    return df_3


def predict(race_infos: pd.DataFrame, seed: int = 10, threshold: float = 0.5) -> None:
    """
    予想する
    """
    race_infos_p = get_infos_p(race_infos)
    target_df = process_categorical(race_infos_p, is_predict=True)

    model_file_name = f"params/model_{seed}.pickle"
    with open(model_file_name, mode="rb") as f:
        lgb_clf = pickle.load(f)

    df = preprocessing_3(lgb_clf, target_df, threshold=threshold)

    predict_list = []

    if df.empty:
        predict_list.append("本日賭けるレースはありません")
        print("本日賭けるレースはありません")
        return

    place_dict = {"01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ",
                  "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"}

    for row in df.iterrows():
        place_id = row[0][6:8]
        place = place_dict[place_id]
        race = row[0][-2:]
        predict = "-".join(map(str, row[1].values))
        predict_str = f"{place}{int(race)}レース {predict}"
        predict_list.append(predict_str)
        print(predict_str)

    return predict_list

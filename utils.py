"""
共通関数群
競争成績URL :https://www1.mbrace.or.jp/od2/K/202201/01/02.html
番組表URL   :https://www1.mbrace.or.jp/od2/B/202201/01/02.html
"""

import json
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import pandas.tseries.offsets as offsets

from results import Results
from infos import Infos
from returns import Returns
from racer import RacerResults
import handle_db


def get_program_list(year: str = "2022", yesterday=False, today=False):
    program_list = []

    if yesterday or today:
        if yesterday:
            dt = datetime.now() - timedelta(1)
        if today:
            dt = datetime.now()
        year = dt.year
        month = dt.month
        day = dt.day
        for place in range(1, 25, 1):
            program_list.append(
                "%s%s/%s/%s" % (year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    else:
        for month in range(1, 13, 1):
            for place in range(1, 25, 1):
                for day in range(1, 32, 1):
                    program_list.append("%s%s/%s/%s" %
                                        (year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    return program_list


def change_format_scrape_data(data):
    data_dict = {}
    for key, value in data.items():
        data_dict[key] = [d.text for d in value]
    return data_dict


def get_scrape_results(program_list: list):
    race_results = Results.scrape(program_list)
    results = change_format_scrape_data(race_results)
    return results


def get_scrape_infos(program_list: list):
    race_infos = Infos.scrape(program_list)
    infos = change_format_scrape_data(race_infos)
    return infos


def save_scrape_data(results: dict = {}, infos: dict = {}, year: str = ""):
    """
    スクレイピングしたデータを保存する
    """
    r = Results(results)
    r.preprocessing()

    i = Infos(infos)
    i.preprocessing()

    rt = Returns(results)
    rt.preprocessing()

    if year:
        results_json_name = "src/results/race_results_%d.json" % year
        infos_json_name = "src/infos/race_infos_%d.json" % year

        results_file_name = "res/results/results_%d.pickle" % year
        infos_file_name = "res/infos/infos_%d.pickle" % year
        returns_file_name = "res/returns/returns_%d.pickle" % year

        with open(results_json_name, "w") as f:
            json.dump(results, f, ensure_ascii=False)
        with open(infos_json_name, "w") as f:
            json.dump(infos, f, ensure_ascii=False)

        r.results_p.to_pickle(results_file_name)
        i.infos_p.to_pickle(infos_file_name)
        rt.returns_p.to_pickle(returns_file_name)

    hdb = handle_db.HandleDB()
    hdb.insert_scrape_data(r.results_p, i.infos_p, rt.returns_p)


def get_latest_date(results_all):
    latest_date = results_all.sort_values(
        "date", ascending=False).head(1)["date"].values[0]
    latest_date = datetime.fromtimestamp(
        latest_date.astype(datetime) * 1e-9).date()
    return latest_date - timedelta(1)


def get_between_date_str(latest_date):
    latest_date = pd.to_datetime(latest_date)
    next_latest_date = latest_date + offsets.Day()
    next_latest_date_str = next_latest_date.strftime('%Y-%m-%d')

    now = datetime.now().date()
    today_str = now.strftime('%Y-%m-%d')

    return today_str, next_latest_date_str


def get_between_program(latest_date):
    today, next_latest_date = get_between_date_str(latest_date)

    today_split = today.split("-")
    today_year = today_split[0]
    today_month = today_split[1]
    today_day = today_split[2]
    latest_split = next_latest_date.split("-")
    latest_year = latest_split[0]
    latest_month = latest_split[1]
    latest_day = latest_split[2]

    program_list = []

    if int(today_month) == int(latest_month):
        print("最新データは今月中")
        for place in range(1, 25, 1):
            for day in range(int(latest_day), int(today) + 1, 1):
                program_list.append(
                    "%s%s/%s/%s" % (latest_year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    elif int(today_month) != int(latest_month):
        print("最新データは先月以前")
        for month in range(int(latest_month), int(today_month) + 1, 1):
            for place in range(1, 25, 1):
                if int(today_month) == int(month):
                    for day in range(1, int(today_day) + 1, 1):
                        program_list.append(
                            "%s%s/%s/%s" % (latest_year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
                else:
                    for day in range(1, 32, 1):
                        program_list.append(
                            "%s%s/%s/%s" % (latest_year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    return program_list


# def update_data(results_all_latest, results_r_latest, returns_latest, race_results, race_infos, between=False):
#     results = change_format_results(race_results)
#     infos = change_format_infos(race_infos)

#     r = Result(results)
#     i = Infos(infos)
#     r.preprocessing()
#     i.preprocessing()

#     results_all_new = r.merge_infos(r.results_p, i.infos_p)

#     rr = RacerResults()
#     n_samples_list = [5, 9, "all"]
#     if between:
#         results_r_new = rr.update_between_data(
#             results_r_latest, results_all_new, n_samples_list)
#     else:
#         results_r_new = rr.update_data(
#             results_all_latest, results_all_new, n_samples_list)

#     rt = Return(results)
#     rt.preprocessing()
#     returns_new = rt.returns

#     results_all = pd.concat([results_all_latest, results_all_new])
#     results_r = pd.concat([results_r_latest, results_r_new])
#     returns = pd.concat([returns_latest, returns_new])

#     results_all.to_pickle("data/results_all.pickle")
#     results_r.to_pickle("data/results_r.pickle")
#     returns.to_pickle("data/returns.pickle")


def process_categorical(results_r, rank, class_int=False, number_del=False, normalize=False):
    df = results_r.copy()
    df["rank"] = df["着順"].map(lambda x: 1 if x <= rank else 0)
    df["艇番"] = df["艇番"].astype("category")
    df["選手番号"] = df["選手番号"].astype("category")
    if number_del:
        df.drop("選手番号", axis=1, inplace=True)
    df.drop("着順", axis=1, inplace=True)
    if not class_int:
        df = pd.get_dummies(df, columns=["class"])
    else:
        class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}
        df["class"] = df["class"].map(class_mapping)
    if normalize:
        def standard_scaler(x): return (x - x.mean()) / x.std(ddof=0)
        for column in tqdm(["全国勝率", "全国2率", "当地勝率", "当地2率", "class",
                            "着順_5R", "着順_1_5R", "着順_2_5R", "着順_3_5R", "着順_4_5R", "着順_5_5R", "着順_6_5R",
                            "着順_9R", "着順_1_9R", "着順_2_9R", "着順_3_9R", "着順_4_9R", "着順_5_9R", "着順_6_9R",
                            "着順_allR", "着順_1_allR", "着順_2_allR", "着順_3_allR", "着順_4_allR", "着順_5_allR", "着順_6_allR"]):
            df[column] = df[column].groupby(level=0).transform(standard_scaler)
    return df


def process_categorical_predict(shusso_df):
    df = shusso_df.copy()
    df["艇番"] = df["艇番"].astype("category")
    df.drop(["date", "選手番号"], axis=1, inplace=True)
    class_mapping = {"B2": 1, "B1": 2, "A2": 3, "A1": 4}
    df["class"] = df["class"].map(class_mapping)
    return df


def predict_proba(model, X):
    def standard_scaler(x):
        return (x - x.mean()) / x.std(ddof=0)

    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    proba = proba.groupby(level=0).transform(standard_scaler)
    proba = (proba - proba.min()) / (proba.max() - proba.min())
    return proba


def rank_join(model, X):
    df = X.copy()[["艇番"]]
    df["point"] = predict_proba(model, X)
    df["rank"] = df["point"].groupby(level=0).rank(ascending=False)
    return df


def pred_table(model, X, threshold=0.5):
    df = rank_join(model, X)
    df["pred_0"] = [0 if p < threshold else 1 for p in df["point"]]
    df["pred_1"] = ((df["pred_0"] == 1) & (df["rank"] == 1)) * 1
    df["pred_2"] = ((df["pred_0"] == 1) & (df["rank"] == 2)) * 1
    df["pred_3"] = ((df["pred_0"] == 1) & (df["rank"] == 3)) * 1
    return df


def preprocessing_3(model, X, threshold=0.5):
    df = pred_table(model, X, threshold)
    df_3_1 = pd.DataFrame(df[df["pred_1"] == 1]["艇番"]
                          ).rename(columns={"艇番": "pred_1"})
    df_3_2 = pd.DataFrame(df[df["pred_2"] == 1]["艇番"]
                          ).rename(columns={"艇番": "pred_2"})
    df_3_3 = pd.DataFrame(df[df["pred_3"] == 1]["艇番"]
                          ).rename(columns={"艇番": "pred_3"})
    df_3_12 = pd.merge(df_3_1, df_3_2, left_index=True,
                       right_index=True, how="right")
    df_3 = pd.merge(df_3_12, df_3_3, left_index=True,
                    right_index=True, how="right")
    df_3["pred_1"] = df_3["pred_1"].astype(int)
    df_3["pred_2"] = df_3["pred_2"].astype(int)
    df_3["pred_3"] = df_3["pred_3"].astype(int)
    return df_3


# def predict(race_infos, results_all):
#     place_dict = {"01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ",
#                   "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"}

#     infos_p = change_format_infos(race_infos)
#     i = Infos(infos_p)
#     i.preprocessing()
#     shusso_df = i.infos_p

#     rr = RacerResults()
#     n_samples_list = [5, 9, "all"]
#     for n_samples in n_samples_list:
#         shusso_df = rr.predict_merge_all(shusso_df, results_all, n_samples)

#     target_df = process_categorical_predict(shusso_df)

#     with open("params/model_3_100_cinn.pickle", mode="rb") as f:
#         lgb_clf = pickle.load(f)

#     df_3t = preprocessing_3(lgb_clf, target_df, threshold=0.665)
#     race_id_list = df_3t.index.unique()
#     predict_list = []
#     for race_id in race_id_list:
#         place_id = race_id[6:8]
#         race = race_id[-2:]
#         place = place_dict[place_id]
#         predict = ("%s%dレース" % (place, int(race)))
#         predict_list.append(predict)

#     df_3t["predict_race"] = predict_list
#     print(df_3t)

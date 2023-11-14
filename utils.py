"""
共通関数群
競争成績URL :https://www1.mbrace.or.jp/od2/K/202201/01/02.html
番組表URL   :https://www1.mbrace.or.jp/od2/B/202201/01/02.html
"""

import json
import pickle
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import pandas.tseries.offsets as offsets
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_o

from results import Results
from infos import Infos
from returns import Returns
from racer import RacerResults
from model import ModelEvaluator
import handle_db

logger = logging.getLogger(__name__)

fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(
    filename="./logs/boatrace.log",
    level=logging.INFO,
    format=fmt,
)


class Log:
    @staticmethod
    def info(str):
        logger.info(str)


def get_program_list(year: str = "2022", yesterday=False, today=False):
    """
    スクレイピングするプログラムリストを返す
    """
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


def get_program_list_debug():
    program_list = []
    for month in range(7, 8, 1):
        for place in range(1, 25, 1):
            for day in range(1, 32, 1):
                program_list.append("2023%s/%s/%s" %
                                    (str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    return program_list


def change_format_scrape_data(data):
    """
    スクレイピングした生データを、json形式に変換して返す
    """
    data_dict = {}
    for key, value in data.items():
        data_dict[key] = [d.text for d in value]
    return data_dict


def get_scrape_results(program_list: list):
    """
    スクレイピングしたレース結果を返す
    """
    race_results = Results.scrape(program_list)
    results = change_format_scrape_data(race_results)
    return results


def get_scrape_infos(program_list: list):
    """
    スクレイピングしたレース情報を返す
    """
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


def get_results_merge_infos():
    """
    resultsとinfosを結合したデータを返す
    """
    hdb = handle_db.HandleDB()
    results_merge_infos = hdb.get_results_all()
    return results_merge_infos


def get_returns():
    """
    returnsを返す
    """
    hdb = handle_db.HandleDB()
    returns = hdb.get_returns()
    return returns


def is_same_latest_data():
    """
    レース結果と払い戻しの最新データが同じかをチェックする
    """
    hdb = handle_db.HandleDB()
    results_latest = hdb.get_results_latest()
    returns_latest = hdb.get_returns_latest()
    if results_latest == returns_latest:
        return True
    else:
        return False


def process_categorical(df, rank, class_int=False, number_del=False):
    """
    カテゴリ変数化、モデル作成前処理を行なったデータを返す
    """
    df["rank"] = df["position"].map(lambda x: 1 if x <= rank else 0)
    df["boat_number"] = df["boat_number"].astype("category")
    df["racer_number"] = df["racer_number"].astype("category")

    if number_del:
        df.drop("racer_number", axis=1, inplace=True)
    if class_int:
        class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}
        df["class"] = df["class"].map(class_mapping)
    else:
        df = pd.get_dummies(df, columns=["class"])

    df.drop("position", axis=1, inplace=True)
    return df


def split_data(df, test_size=0.3):
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


def get_train_valid(results_c):
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


def get_lgb_train_valid(results):
    """
    optuna用にデータを用意
    """
    X_train, y_train, X_valid, y_valid = get_train_valid(results)

    lgb_train = lgb_o.Dataset(X_train.values, y_train.values)
    lgb_valid = lgb_o.Dataset(X_valid.values, y_valid.values)

    params = {
        "objective": "binary",
        "random_state": 100,
        "metric": "auc",
    }
    return lgb_train, lgb_valid, params


def get_train_test(results_c):
    """
    訓練データとテストデータを返す
    """
    train, test = split_data(results_c)
    X_train = train.drop(["rank", "date"], axis=1)
    y_train = train["rank"]
    X_test = test.drop(["rank", "date"], axis=1)
    y_test = test["rank"]
    return X_train, y_train, X_test, y_test


def get_optuna_params(params, lgb_train, lgb_valid, seed):
    """
    パラメータチューニングを行なった結果を返す
    """
    lgb_clf_o = lgb_o.train(
        params, lgb_train,
        valid_sets=(lgb_train, lgb_valid),
        verbose_eval=-1,
        early_stopping_rounds=10,
        optuna_seed=seed
    )
    params_o = lgb_clf_o.params
    del params_o["early_stopping_round"]
    return params_o


def get_lgb_clf(params_o, X_train, y_train):
    """
    モデルを作成し、返す
    """
    lgb_clf = lgb.LGBMClassifier(**params_o)
    lgb_clf.fit(X_train.values, y_train.values)
    return lgb_clf


def gain(return_func, n_samples=100):
    """
    閾値ごとの回収率を算出する
    """
    gain = {}
    try:
        for i in tqdm(range(n_samples)):
            threshold = 0.5 + (((1 - 0.5) / n_samples) * i)
            n_bets, return_rate = return_func(threshold)
            gain[threshold] = {"n_bets": n_bets,
                               "return_rate": return_rate}
        return pd.DataFrame(gain).T
    except:
        return pd.DataFrame(gain).T


def get_gain_dict(lgb_clf, returns, X_test):
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
    gain_2t_n = gain(me.nirentan_nagashi)
    gain_2t_b = gain(me.nirentan_box)
    gain_3t_2 = gain(me.sanrentan_nagashi_2)
    gain_3t_3 = gain(me.sanrentan_nagashi_3)
    gain_3t_b = gain(me.sanrentan_box)
    gain_dict = {"gain_t": gain_t, "gain_f": gain_f, "gain_2t": gain_2t, "gain_2f": gain_2f, "gain_3t": gain_3t, "gain_3f": gain_3f,
                 "gain_2t_n": gain_2t_n, "gain_2t_b": gain_2t_b, "gain_3t_2": gain_3t_2, "gain_3t_3": gain_3t_3, "gain_3t_b": gain_3t_b}
    return gain_dict


def save_model(gain_dict, model, rank, class_int, number_del, seed):
    """
    モデルとパラメータを保存する
    """
    params = {}
    str_ci = ""
    str_nd = ""

    for k, v in gain_dict.items():
        return_max = v[v["n_bets"] > 200].sort_values(
            "return_rate", ascending=False).head(1)
        max_return_rate = return_max.iloc[0]["return_rate"]
        if max_return_rate > 110:
            params[k] = {}
            params[k]["threshold"] = return_max.index[0]
            params[k]["return_rate"] = max_return_rate

    print(params)

    if params:
        if class_int:
            str_ci = "_ci"
        if number_del:
            str_nd = "_nd"

        file_name = "params/model_%d_%d%s%s" % (rank, seed, str_ci, str_nd)
        print(file_name)

        model_file_name = file_name + ".pickle"
        with open(model_file_name, mode="wb") as f:
            pickle.dump(model, f)

        json_file_name = file_name + ".json"
        with open(json_file_name, "w") as f:
            json.dump(params, f, ensure_ascii=False)


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


def process_categorical_predict(df):
    """
    カテゴリ変数化、前処理を行なった予想に用いるデータを返す
    """
    df["boat_number"] = df["boat_number"].astype("category")
    df["racer_number"] = df["racer_number"].astype("category")
    df.set_index("race_id", inplace=True)
    class_mapping = {"B2": 1, "B2": 2, "A2": 3, "A1": 4}
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
    df = X.copy()[["boat_number"]]
    df["point"] = predict_proba(model, X)
    df["rank"] = df["point"].groupby(level=0).rank(ascending=False)
    return df


def pred_table(model, target_df, threshold=0.5):
    df = rank_join(model, target_df)
    df["pred_0"] = [0 if p < threshold else 1 for p in df["point"]]
    df["pred_1"] = ((df["pred_0"] == 1) & (df["rank"] == 1)) * 1
    df["pred_2"] = ((df["pred_0"] == 1) & (df["rank"] == 2)) * 1
    df["pred_3"] = ((df["pred_0"] == 1) & (df["rank"] == 3)) * 1
    return df


def preprocessing_3(model, target_df, threshold):
    """
    3連単の予想データを返す
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


def predict(race_infos):
    """
    予想する
    """
    i = Infos(race_infos)
    i.preprocessing()

    shusso_df = i.infos_p
    target_df = process_categorical_predict(shusso_df)

    with open("params/model_2_10_naci.pickle", mode="rb") as f:
        lgb_clf = pickle.load(f)

    df = preprocessing_3(lgb_clf, target_df, threshold=0.66)

    race_id_list = df.index.unique()
    predict_list = []
    place_dict = {"01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ",
                  "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"}

    for race_id in race_id_list:
        place_id = race_id[6:8]
        race = race_id[-2:]
        place = place_dict[place_id]
        predict = ("%s%dレース" % (place, int(race)))
        predict_list.append(predict)
    df["predict_race"] = predict_list
    print(df)


def check_model(returns, X):
    """
    モデルの回収率を確認する
    """
    with open("params/model_2_10_naci.pickle", mode="rb") as f:
        lgb_clf = pickle.load(f)

    me = ModelEvaluator(lgb_clf, returns, X)
    Log.info(me.score)

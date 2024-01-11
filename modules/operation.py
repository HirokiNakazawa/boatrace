"""
運用関連の関数群
"""

from modules.predict import *
from tqdm import tqdm


def profit(infos: pd.DataFrame, returns: pd.DataFrame, seed: int = 10, threshold: float = 0.5) -> None:
    """
    収益を算出する
    """
    model_file_name = f"params/model_{seed}.pickle"
    with open(model_file_name, mode="rb") as f:
        lgb_clf = pickle.load(f)

    infos_o = infos[infos["date"] >= "2024-01-01"].copy()

    dfs_by_date = [group for _, group in infos_o.groupby(
        infos_o['date'].dt.date)]
    df_3s = []
    for df_by_date in tqdm(dfs_by_date):
        target_df = process_categorical(df_by_date, is_predict=True)
        df_3 = preprocessing_3(lgb_clf, target_df, threshold=0.67)
        df_3s.append(df_3)

    df_3 = pd.concat([data for data in df_3s])
    df = pd.merge(df_3, returns[["win_3_1", "win_3_2", "win_3_3",
                  "return_3t"]], left_index=True, right_index=True, how="left")

    df_hits = df[(df["pred_1"] == df["win_3_1"]) & (df["pred_2"] == df["win_3_2"]) & (df["pred_3"] == df["win_3_3"]) |
                 (df["pred_1"] == df["win_3_1"]) & (df["pred_2"] == df["win_3_3"]) & (df["pred_3"] == df["win_3_2"]) |
                 (df["pred_1"] == df["win_3_2"]) & (df["pred_2"] == df["win_3_1"]) & (df["pred_3"] == df["win_3_3"]) |
                 (df["pred_1"] == df["win_3_2"]) & (df["pred_2"] == df["win_3_3"]) & (df["pred_3"] == df["win_3_1"]) |
                 (df["pred_1"] == df["win_3_3"]) & (df["pred_2"] == df["win_3_1"]) & (df["pred_3"] == df["win_3_2"]) |
                 (df["pred_1"] == df["win_3_3"]) & (df["pred_2"] == df["win_3_2"]) & (df["pred_3"] == df["win_3_1"])]

    use_money = len(df) * 600
    return_money = sum(df_hits["return_3t"])
    profit = return_money - use_money
    print(f"収益: {profit} 円")

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

    target_df = process_categorical(infos_o, is_predict=True)
    df_p = preprocessing(lgb_clf, target_df, threshold=threshold)
    df_p = df_p[df_p["pred_3"] != 0]
    df = pd.merge(df_p, returns[["win_1", "win_2", "win_3", "return_3t"]],
                  left_index=True, right_index=True, how="left")

    df_hits = df[(((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                  ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))) &
                 (df["pred_3"] == df["win_3"])]

    use_money = len(df) * 200
    return_money = sum(df_hits["return_3t"])
    profit = return_money - use_money
    print(f"収益: {profit} 円")

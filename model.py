import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class ModelEvaluator:
    def __init__(self, model: lgb.LGBMClassifier, returns: pd.DataFrame, X: pd.DataFrame) -> None:
        self.model = model
        self.returns = returns
        self.rank_join_df = self.rank_join(X)

    def predict_proba(self, X: pd.DataFrame, std: bool = True, minmax: bool = True) -> pd.Series:
        proba = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        if std:
            proba = proba.groupby(level=0).transform(
                lambda x: (x - x.mean()) / x.std(ddof=0))
        if minmax:
            scaler = MinMaxScaler()
            proba = proba.groupby(level=0).transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
        return proba

    def rank_join(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()[["boat_number"]]
        df["point"] = self.predict_proba(X)
        df["rank"] = df["point"].groupby(level=0).rank(ascending=False)
        return df

    def score(self, y_true: pd.DataFrame, X: pd.DataFrame) -> float:
        return roc_auc_score(y_true, self.predict_proba(X))

    def feature_importance(self, X: pd.DataFrame, n_display: int = 20) -> pd.DataFrame:
        importances = pd.DataFrame({"features": X.columns,
                                   "importance": self.model.feature_importances_})
        return importances.sort_values("importance", ascending=False)[:n_display]

    def pred_table(self, threshold: float = 0.5) -> pd.DataFrame:
        df = self.rank_join_df.copy()
        df["pred_1"] = ((df["point"] > threshold) & (df["rank"] == 1)) * 1
        df["pred_2"] = ((df["point"] > threshold) & (df["rank"] == 2)) * 1
        df["pred_3"] = ((df["point"] > threshold) & (df["rank"] == 3)) * 1
        return df

    def preprocessing(self, threshold: float = 0.5) -> pd.DataFrame:
        df = self.pred_table(threshold)
        df_p = pd.DataFrame()
        df["boat_number"] = df["boat_number"].astype(int)
        df_p["pred_1"] = pd.DataFrame(df[df["pred_1"] == 1]["boat_number"])
        df_p["pred_2"] = pd.DataFrame(df[df["pred_2"] == 1]["boat_number"])
        df_p["pred_3"] = pd.DataFrame(df[df["pred_3"] == 1]["boat_number"])
        df_p.fillna(0, inplace=True)
        df_m = pd.merge(df_p, self.returns, left_index=True,
                        right_index=True, how="left")
        return df_m

    def hits(self, df: pd.DataFrame, kind: str = "") -> Tuple[pd.DataFrame, int, int, float, int] | None:
        n_bets = len(df)
        if kind == "nirentan":
            df_hits = df[(df["pred_1"] == df["win_1"]) &
                         (df["pred_2"] == df["win_2"])]
            money = sum(df_hits["return_2t"])
            use_money = n_bets * 100
        elif kind == "nirentan_nagashi":
            df_hits = df[df["pred_1"] == df["win_1"]]
            money = sum(df_hits["return_2t"])
            use_money = n_bets * 500
        elif kind == "nirentan_box":
            df_hits = df[((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))]
            money = sum(df_hits["return_2t"])
            use_money = n_bets * 200
        elif kind == "nirenpuku":
            n_bets = len(df)
            df_hits = df[((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))]
            money = sum(df_hits["return_2f"])
            use_money = n_bets * 100
        elif kind == "sanrentan":
            df_hits = df[(df["pred_1"] == df["win_1"]) & (
                df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_3"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 100
        elif kind == "sanrentan_nagashi_1":
            df_hits = df[df["pred_1"] == df["win_1"]]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 2000
        elif kind == "sanrentan_nagashi_2":
            df_hits = df[(df["pred_1"] == df["win_1"]) &
                         (df["pred_2"] == df["win_2"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 400
        elif kind == "sanrentan_12_box":
            df_hits = df[(((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))) &
                         (df["pred_3"] == df["win_3"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 200
        elif kind == "sanrentan_12_box_nagashi":
            df_hits = df[((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 800
        elif kind == "sanrentan_box":
            df_hits = df[(df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_1"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_1"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 600
        elif kind == "sanrenpuku":
            df_hits = df[(df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_1"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_1"])]
            money = sum(df_hits["return_3f"])
            use_money = n_bets * 100
        else:
            print("kind is not found")
            return
        return_rate = (money / use_money) * 100
        return_money = money - use_money
        return df_hits, n_bets, use_money, return_rate, return_money

    def nirentan_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirentan")

    def nirentan_nagashi(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirentan_nagashi")

    def nirentan_box(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirentan_box")

    def nirenpuku_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirenpuku")

    def sanrentan_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan")

    def sanrentan_nagashi_1(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_nagashi_1")

    def sanrentan_nagashi_2(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_nagashi_2")

    def sanrentan_12_box(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_12_box")

    def sanrentan_12_box_nagashi(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_12_box_nagashi")

    def sanrentan_box(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_box")

    def sanrenpuku_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float, int]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrenpuku")

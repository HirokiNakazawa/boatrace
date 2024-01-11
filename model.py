import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from typing import Tuple


class ModelEvaluator:
    def __init__(self, model: lgb.LGBMClassifier, returns: pd.DataFrame, X: pd.DataFrame) -> None:
        self.model = model
        self.returns = returns
        self.rank_join_df = self.rank_join(X)

    def predict_proba(self, X: pd.DataFrame, std: bool = True, minmax: bool = True) -> pd.Series:
        proba = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        if std:
            def standard_scaler(x): return (x - x.mean()) / x.std(ddof=0)
            proba = proba.groupby(level=0).transform(standard_scaler)
        if minmax:
            proba = (proba - proba.min()) / (proba.max() - proba.min())
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

    def hits(self, df: pd.DataFrame, kind: str = "") -> Tuple[pd.DataFrame, int, int, float] | None:
        n_bets = len(df)
        if kind == "tansho":
            df_hits = df[df["pred_1"] == df["win_1"]]
            money = sum(df_hits["return_t"])
            use_money = n_bets * 100
            return_rate = (money / use_money) * 100
        elif kind == "fukusho":
            df_hits = df[(df["pred_1"] == df["win_1"]) |
                         (df["pred_1"] == df["win_2"])]
            money = sum(df[df["pred_1"] == df["win_1"]]["return_f1"]) + \
                sum(df[df["pred_1"] == df["win_2"]]["return_f2"])
            use_money = n_bets * 100
            return_rate = (money / use_money) * 100
        elif kind == "nirentan":
            df_hits = df[(df["pred_1"] == df["win_1"]) &
                         (df["pred_2"] == df["win_2"])]
            money = sum(df_hits["return_2t"])
            use_money = n_bets * 100
            return_rate = (money / use_money) * 100
        elif kind == "nirentan_nagashi":
            df_hits = df[df["pred_1"] == df["win_1"]]
            money = sum(df_hits["return_2t"])
            use_money = n_bets * 500
            return_rate = (money / use_money) * 100
        elif kind == "nirentan_box":
            df_hits = df[((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))]
            money = sum(df_hits["return_2t"])
            use_money = n_bets * 200
            return_rate = (money / use_money) * 100
        elif kind == "nirenpuku":
            n_bets = len(df)
            df_hits = df[((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))]
            money = sum(df_hits["return_2f"])
            use_money = n_bets * 100
            return_rate = (money / use_money) * 100
        elif kind == "sanrentan":
            df_hits = df[(df["pred_1"] == df["win_1"]) & (
                df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_3"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 100
            return_rate = (money / use_money) * 100
        elif kind == "sanrentan_nagashi_1":
            df_hits = df[df["pred_1"] == df["win_1"]]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 2000
            return_rate = (money / use_money) * 100
        elif kind == "sanrentan_nagashi_2":
            df_hits = df[(df["pred_1"] == df["win_1"]) &
                         (df["pred_2"] == df["win_2"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 400
            return_rate = (money / use_money) * 100
        elif kind == "sanrentan_12_box":
            df_hits = df[(((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))) &
                         (df["pred_3"] == df["win_3"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 200
            return_rate = (money / use_money) * 100
        elif kind == "sanrentan_12_box_nagashi":
            df_hits = df[((df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"])) |
                         ((df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]))]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 800
            return_rate = (money / use_money) * 100
        elif kind == "sanrentan_box":
            df_hits = df[(df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_1"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_1"])]
            money = sum(df_hits["return_3t"])
            use_money = n_bets * 600
            return_rate = (money / use_money) * 100
        elif kind == "sanrenpuku":
            df_hits = df[(df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_1"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_3"]) |
                         (df["pred_1"] == df["win_2"]) & (df["pred_2"] == df["win_3"]) & (df["pred_3"] == df["win_1"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_1"]) & (df["pred_3"] == df["win_2"]) |
                         (df["pred_1"] == df["win_3"]) & (df["pred_2"] == df["win_2"]) & (df["pred_3"] == df["win_1"])]
            money = sum(df_hits["return_3f"])
            use_money = n_bets * 100
            return_rate = (money / use_money) * 100
        else:
            print("kind is not found")
            return
        return df_hits, n_bets, use_money, return_rate

    def tansho_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        return self.hits(df, kind="tansho")

    def fukusho_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        return self.hits(df, kind="fukusho")

    def nirentan_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirentan")

    def nirentan_nagashi(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirentan_nagashi")

    def nirentan_box(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirentan_box")

    def nirenpuku_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_2"] != 0]
        return self.hits(df, kind="nirenpuku")

    def sanrentan_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan")

    def sanrentan_nagashi_1(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_nagashi_1")

    def sanrentan_nagashi_2(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_nagashi_2")

    def sanrentan_12_box(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_12_box")

    def sanrentan_12_box_nagashi(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_12_box_nagashi")

    def sanrentan_box(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrentan_box")

    def sanrenpuku_return(self, threshold: float = 0.5) -> Tuple[pd.DataFrame, int, int, float]:
        df = self.preprocessing(threshold)
        df = df[df["pred_3"] != 0]
        return self.hits(df, kind="sanrenpuku")

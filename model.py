import pandas as pd
from sklearn.metrics import roc_auc_score


class ModelEvaluator:
    def __init__(self, model, returns, X):
        self.model = model
        self.tansho = returns[["win_t", "return_t"]]
        self.fukusho = returns[["win_f1", "return_f1", "win_f2", "return_f2"]]
        self.nirentan = returns[["win_2t_1", "win_2t_2", "return_2t"]]
        self.nirenpuku = returns[["win_2f_1", "win_2f_2", "return_2f"]]
        self.sanrentan = returns[["win_3t_1",
                                  "win_3t_2", "win_3t_3", "return_3t"]]
        self.sanrenpuku = returns[["win_3f_1",
                                   "win_3f_2", "win_3f_3", "return_3f"]]
        self.rank_join_df = self.rank_join(X)

    def predict_proba(self, X, std=True, minmax=True):
        proba = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        if std:
            def standard_scaler(x): return (x - x.mean()) / x.std(ddof=0)
            proba = proba.groupby(level=0).transform(standard_scaler)
        if minmax:
            proba = (proba - proba.min()) / (proba.max() - proba.min())
        return proba

    def rank_join(self, X):
        df = X.copy()[["boat_number"]]
        df["point"] = self.predict_proba(X)
        df["rank"] = df["point"].groupby(level=0).rank(ascending=False)
        return df

    def score(self, y_true, X):
        return roc_auc_score(y_true, self.predict_proba(X))

    def feature_importance(self, X, n_display=20):
        importances = pd.DataFrame({"features": X.columns,
                                   "importance": self.model.feature_importances_})
        return importances.sort_values("importance", ascending=False)[:n_display]

    def pred_table(self, threshold=0.5):
        df = self.rank_join_df.copy()
        df["pred_0"] = [0 if p < threshold else 1 for p in df["point"]]
        df["pred_1"] = ((df["pred_0"] == 1) & (df["rank"] == 1)) * 1
        df["pred_2"] = ((df["pred_0"] == 1) & (df["rank"] == 2)) * 1
        df["pred_3"] = ((df["pred_0"] == 1) & (df["rank"] == 3)) * 1
        return df

    def preprocessing_2(self, pred_table):
        df = pred_table
        df_2_1 = pd.DataFrame(df[df["pred_1"] == 1]["boat_number"]).rename(
            columns={"boat_number": "pred_1"})
        df_2_2 = pd.DataFrame(df[df["pred_2"] == 1]["boat_number"]).rename(
            columns={"boat_number": "pred_2"})
        df_2 = pd.merge(df_2_1, df_2_2, left_index=True,
                        right_index=True, how="right")
        df_2["pred_1"] = df_2["pred_1"].astype(int)
        df_2["pred_2"] = df_2["pred_2"].astype(int)
        return df_2

    def preprocessing_3(self, pred_table):
        df = pred_table
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

    def hits(self, df, kind=""):
        n_bets = len(df)
        if kind == "tansho":
            df_hits = df[df["boat_number"] == df["win_t"]]
            df_median = df_hits["return_t"].median()
            money = sum(df_hits["return_t"])
            median_money = df_median * len(df_hits)
        elif kind == "fukusho":
            df_hits = df[(df["boat_number"] == df["win_f1"]) |
                         (df["boat_number"] == df["win_f2"])]
            df_hits_1 = df_hits[df_hits["boat_number"] == df_hits["win_f1"]]
            df_hits_2 = df_hits[df_hits["boat_number"] == df_hits["win_f2"]]
            df_median_1 = df_hits_1["return_f1"].median()
            df_median_2 = df_hits_2["return_f2"].median()
            money = sum(df[df["boat_number"] == df["win_f1"]]["return_f1"]) + \
                sum(df[df["boat_number"] == df["win_f2"]]["return_f2"])
            median_money = df_median_1 * \
                len(df_hits_1) + df_median_2 * len(df_hits_2)
        elif kind == "nirentan":
            df_hits = df[(df["pred_1"] == df["win_2t_1"]) &
                         (df["pred_2"] == df["win_2t_2"])]
            df_median = df_hits["return_2t"].median()
            money = sum(df_hits["return_2t"])
            median_money = df_median * len(df_hits)
        elif kind == "nirenpuku":
            df_hits = df[((df["pred_1"] == df["win_2f_1"]) & (df["pred_2"] == df["win_2f_2"])) |
                         ((df["pred_1"] == df["win_2f_2"]) & (df["pred_2"] == df["win_2f_1"]))]
            df_median = df_hits["return_2f"].median()
            money = sum(df_hits["return_2f"])
            median_money = df_median * len(df_hits)
        elif kind == "sanrentan":
            df_hits = df[(df["pred_1"] == df["win_3t_1"]) & (
                df["pred_2"] == df["win_3t_2"]) & (df["pred_3"] == df["win_3t_3"])]
            df_median = df_hits["return_3t"].median()
            money = sum(df_hits["return_3t"])
            median_money = df_median * len(df_hits)
        elif kind == "sanrenpuku":
            df_hits = df[(df["pred_1"] == df["win_3f_1"]) & (df["pred_2"] == df["win_3f_2"]) & (df["pred_3"] == df["win_3f_3"]) |
                         (df["pred_1"] == df["win_3f_1"]) & (df["pred_2"] == df["win_3f_3"]) & (df["pred_3"] == df["win_3f_2"]) |
                         (df["pred_1"] == df["win_3f_2"]) & (df["pred_2"] == df["win_3f_1"]) & (df["pred_3"] == df["win_3f_3"]) |
                         (df["pred_1"] == df["win_3f_2"]) & (df["pred_2"] == df["win_3f_3"]) & (df["pred_3"] == df["win_3f_1"]) |
                         (df["pred_1"] == df["win_3f_3"]) & (df["pred_2"] == df["win_3f_1"]) & (df["pred_3"] == df["win_3f_2"]) |
                         (df["pred_1"] == df["win_3f_3"]) & (df["pred_2"] == df["win_3f_2"]) & (df["pred_3"] == df["win_3f_1"])]
            df_median = df_hits["return_3f"].median()
            money = sum(df_hits["return_3f"])
            median_money = df_median * len(df_hits)
        else:
            print("kind is not found")
            return
        return_rate = (money / (n_bets * 100)) * 100
        return_median = (median_money / (n_bets * 100)) * 100
        return df_hits, n_bets, return_rate, return_median

    def tansho_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = pred_table.copy()
        df = df[df["pred_0"] == 1]
        df["boat_number"] = df["boat_number"].astype(int)
        df = pd.merge(df, self.tansho, how="left", on="race_id")
        return self.hits(df, kind="tansho")

    def fukusho_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = pred_table.copy()
        df = df[df["pred_0"] == 1]
        df["boat_number"] = df["boat_number"].astype(int)
        df = pd.merge(df, self.fukusho, how="left", on="race_id")
        return self.hits(df, kind="fukusho")

    def nirentan_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_2(pred_table)
        df = pd.merge(df, self.nirentan, how="left", on="race_id")
        return self.hits(df, kind="nirentan")

    def nirenpuku_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_2(pred_table)
        df = pd.merge(df, self.nirenpuku, how="left", on="race_id")
        return self.hits(df, kind="nirenpuku")

    def sanrentan_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        df = pd.merge(df, self.sanrentan, how="left", on="race_id")
        return self.hits(df, kind="sanrentan")

    def sanrenpuku_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        df = pd.merge(df, self.sanrenpuku, how="left", on="race_id")
        return self.hits(df, kind="sanrenpuku")

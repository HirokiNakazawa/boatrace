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
        df = X.copy()[["艇番"]]
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
        df_2_1 = pd.DataFrame(df[df["pred_1"] == 1]["艇番"]).rename(
            columns={"艇番": "pred_1"})
        df_2_2 = pd.DataFrame(df[df["pred_2"] == 1]["艇番"]).rename(
            columns={"艇番": "pred_2"})
        df_2 = pd.merge(df_2_1, df_2_2, left_index=True,
                        right_index=True, how="right")
        df_2["pred_1"] = df_2["pred_1"].astype(int)
        df_2["pred_2"] = df_2["pred_2"].astype(int)
        return df_2

    def preprocessing_3(self, pred_table):
        df = pred_table
        df_3_1 = pd.DataFrame(df[df["pred_1"] == 1]["艇番"]).rename(
            columns={"艇番": "pred_1"})
        df_3_2 = pd.DataFrame(df[df["pred_2"] == 1]["艇番"]).rename(
            columns={"艇番": "pred_2"})
        df_3_3 = pd.DataFrame(df[df["pred_3"] == 1]["艇番"]).rename(
            columns={"艇番": "pred_3"})
        df_3_12 = pd.merge(df_3_1, df_3_2, left_index=True,
                           right_index=True, how="right")
        df_3 = pd.merge(df_3_12, df_3_3, left_index=True,
                        right_index=True, how="right")
        df_3["pred_1"] = df_3["pred_1"].astype(int)
        df_3["pred_2"] = df_3["pred_2"].astype(int)
        df_3["pred_3"] = df_3["pred_3"].astype(int)
        return df_3

    def tansho_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = pred_table.copy()
        df = df[df["pred_0"] == 1]
        df["艇番"] = df["艇番"].astype(int)
        n_bets = len(df)
        df = pd.merge(df, self.tansho, how="left", on="race_id")
        money = sum(df[df["艇番"] == df["win_t"]]["return_t"])
        return_rate = (money / (n_bets * 100)) * 100
        return n_bets, return_rate

    def fukusho_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = pred_table.copy()
        df = df[df["pred_0"] == 1]
        df["艇番"] = df["艇番"].astype(int)
        n_bets = len(df)
        df = pd.merge(df, self.fukusho, how="left", on="race_id")
        money = sum(df[df["艇番"] == df["win_f1"]]["return_f1"]) +\
            sum(df[df["艇番"] == df["win_f2"]]["return_f2"])
        return_rate = (money / (n_bets * 100)) * 100
        return n_bets, return_rate

    def nirentan_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_2(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.nirentan, how="left", on="race_id")
        money = sum(df[(df["pred_1"] == df["win_2t_1"]) & (
            df["pred_2"] == df["win_2t_2"])]["return_2t"])
        return_rate = (money / (n_bets * 100)) * 100
        return n_bets, return_rate

    def nirentan_nagashi(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_2(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.nirentan, how="left", on="race_id")
        money = sum(df[df["pred_1"] == df["win_2t_1"]]["return_2t"])
        return_rate = (money / (n_bets * 500)) * 100
        return n_bets, return_rate

    def nirentan_box(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_2(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.nirentan, how="left", on="race_id")
        money = sum(df[(df["pred_1"] == df["win_2t_1"]) & (df["pred_2"] == df["win_2t_2"])]["return_2t"]) +\
            sum(df[(df["pred_1"] == df["win_2t_2"]) & (
                df["pred_2"] == df["win_2t_1"])]["return_2t"])
        return_rate = (money / (n_bets * 200)) * 100
        return n_bets, return_rate

    def nirenpuku_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_2(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.nirenpuku, how="left", on="race_id")
        money = sum(df[((df["pred_1"] == df["win_2f_1"]) & (df["pred_2"] == df["win_2f_2"]))]["return_2f"]) +\
            sum(df[((df["pred_1"] == df["win_2f_2"]) & (
                df["pred_2"] == df["win_2f_1"]))]["return_2f"])
        return_rate = (money / (n_bets * 100)) * 100
        return n_bets, return_rate

    def sanrentan_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.sanrentan, how="left", on="race_id")
        money = sum(df[(df["pred_1"] == df["win_3t_1"]) & (df["pred_2"] == df["win_3t_2"]) &
                       (df["pred_3"] == df["win_3t_3"])]["return_3t"])
        return_rate = (money / (n_bets * 100)) * 100
        return n_bets, return_rate

    def sanrentan_nagashi_2(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.sanrentan, how="left", on="race_id")
        money = sum(df[df["pred_1"] == df["win_3t_1"]]["return_3t"])
        return_rate = (money / (n_bets * 2000)) * 100
        return n_bets, return_rate

    def sanrentan_nagashi_3(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.sanrentan, how="left", on="race_id")
        money = sum(df[(df["pred_1"] == df["win_3t_1"]) & (
            df["pred_2"] == df["win_3t_2"])]["return_3t"])
        return_rate = (money / (n_bets * 400)) * 100
        return n_bets, return_rate

    def sanrentan_box(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.sanrentan, how="left", on="race_id")
        money = sum(df[(df["pred_1"] == df["win_3t_1"]) & (df["pred_2"] == df["win_3t_2"]) &
                       (df["pred_3"] == df["win_3t_3"])]["return_3t"]) +\
            sum(df[(df["pred_1"] == df["win_3t_1"]) & (df["pred_2"] == df["win_3t_3"]) &
                   (df["pred_3"] == df["win_3t_2"])]["return_3t"]) +\
            sum(df[(df["pred_1"] == df["win_3t_2"]) & (df["pred_2"] == df["win_3t_1"]) &
                   (df["pred_3"] == df["win_3t_3"])]["return_3t"]) +\
            sum(df[(df["pred_1"] == df["win_3t_2"]) & (df["pred_2"] == df["win_3t_3"]) &
                   (df["pred_3"] == df["win_3t_1"])]["return_3t"]) +\
            sum(df[(df["pred_1"] == df["win_3t_3"]) & (df["pred_2"] == df["win_3t_1"]) &
                   (df["pred_3"] == df["win_3t_2"])]["return_3t"]) +\
            sum(df[(df["pred_1"] == df["win_3t_3"]) & (df["pred_2"] == df["win_3t_2"]) &
                   (df["pred_3"] == df["win_3t_1"])]["return_3t"])
        return_rate = (money / (n_bets * 600)) * 100
        return n_bets, return_rate

    def sanrenpuku_return(self, threshold=0.5):
        pred_table = self.pred_table(threshold)
        df = self.preprocessing_3(pred_table)
        n_bets = len(df)
        df = pd.merge(df, self.sanrenpuku, how="left", on="race_id")
        money = sum(df[(df["pred_1"] == df["win_3f_1"]) & (df["pred_2"] == df["win_3f_2"]) &
                       (df["pred_3"] == df["win_3f_3"])]["return_3f"]) +\
            sum(df[(df["pred_1"] == df["win_3f_1"]) & (df["pred_2"] == df["win_3f_3"]) &
                   (df["pred_3"] == df["win_3f_2"])]["return_3f"]) +\
            sum(df[(df["pred_1"] == df["win_3f_2"]) & (df["pred_2"] == df["win_3f_1"]) &
                   (df["pred_3"] == df["win_3f_3"])]["return_3f"]) +\
            sum(df[(df["pred_1"] == df["win_3f_2"]) & (df["pred_2"] == df["win_3f_3"]) &
                   (df["pred_3"] == df["win_3f_1"])]["return_3f"]) +\
            sum(df[(df["pred_1"] == df["win_3f_3"]) & (df["pred_2"] == df["win_3f_1"]) &
                   (df["pred_3"] == df["win_3f_2"])]["return_3f"]) +\
            sum(df[(df["pred_1"] == df["win_3f_3"]) & (df["pred_2"] == df["win_3f_2"]) &
                   (df["pred_3"] == df["win_3f_1"])]["return_3f"])
        return_rate = (money / (n_bets * 100)) * 100
        return n_bets, return_rate

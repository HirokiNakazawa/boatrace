import pandas as pd
import numpy as np
from tqdm import tqdm


class RacerResults:
    def __init__(self):
        pass

    def average(self, results_all_old, n_samples_list):
        def get_filtered_df(df):
            if n_samples == "all":
                filtered_df = df
            else:
                filtered_df = df.sort_values(
                    "date", ascending=False).groupby(level=0).head(n_samples)
            return filtered_df

        target_df = results_all_old.copy()
        target_df = target_df.set_index(["選手番号"])

        averages = []
        for n_samples in n_samples_list:
            average = get_filtered_df(target_df).groupby(
                level=0)[["着順"]].mean().add_suffix("_{}R".format(n_samples))
            averages.append(average)

            for i in range(1, 7):
                target_df_number = target_df[target_df["艇番"] == str(i)]
                average_number = get_filtered_df(target_df_number).groupby(
                    level=0)[["着順"]].mean().add_suffix("_{}_{}R".format(i, n_samples))
                averages.append(average_number)

        return averages

    def update_merge(self, results, results_all_old, date, n_samples_list):
        df = results[results["date"] == date]
        averages = self.average(results_all_old, n_samples_list)
        for average in averages:
            df = df.merge(average, left_on="選手番号",
                          right_index=True, how="left")
        return df

    def update_data(self, results_all_old, results_all_new, n_samples_list):
        results = results_all_new.copy()
        date_list = results["date"].unique()
        merged_all_df = pd.concat([self.update_merge(
            results, results_all_old, date, n_samples_list) for date in tqdm(date_list)])
        return merged_all_df

    def update_between_data(self, results_r_old, results_all_new, n_samples_list):
        results = results_all_new.copy()
        results_old = results_r_old.copy()
        date_list = np.unique(results_all_new["date"].sort_values().values)
        for date in tqdm(date_list):
            merged_df = self.update_merge(
                results, results_old, date, n_samples_list)
            results_old = pd.concat([results_old, merged_df])
        df = results_old.copy()
        return df

    def predict_average(self, results, date, n_samples="all"):
        target_df = results.copy()
        target_df = target_df.set_index(["選手番号"])

        if n_samples == "all":
            filtered_df = target_df[target_df["date"] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df["date"] < date].sort_values("date", ascending=False).\
                groupby(level=0).head(n_samples)
        else:
            raise Exception("n_samples must be > 0")

        average_1 = filtered_df.groupby(level=0)[["着順"]].mean()
        average_2 = filtered_df.groupby(["選手番号", "艇番"])[
            ["着順"]].mean().unstack()
        average_2.columns = average_2.columns.map(lambda x: "_".join(x))

        average = pd.concat([average_1, average_2], axis=1).add_suffix(
            "_{}R".format(n_samples))
        return average

    def predict_merge(self, infos, results_all_old, date, n_samples="all"):
        df = infos[infos["date"] == date]
        results = results_all_old.copy()
        merged_df = df.merge(self.predict_average(results, date, n_samples),
                             left_on="選手番号", right_index=True, how="left")
        return merged_df

    def predict_merge_all(self, infos_new, results_all_old, n_samples="all"):
        infos = infos_new.copy()
        date_list = infos["date"].unique()
        merged_all_df = pd.concat([self.predict_merge(
            infos, results_all_old, date, n_samples) for date in tqdm(date_list)])
        return merged_all_df

from tqdm import tqdm
import pandas as pd
import datetime


class RacerResults:
    def __init__(self, racer_results: pd.DataFrame) -> None:
        self.racer_results = racer_results

    def average(self, date: datetime.date, n_samples: int | str = "all") -> pd.DataFrame:
        if n_samples == "all":
            filtered_df = self.racer_results[self.racer_results["date"] < date]
        elif n_samples > 0:
            filtered_df = self.racer_results[self.racer_results["date"] < date].sort_values(
                "date", ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception("n_samples must be > 0")

        average_df = filtered_df[["position"]].groupby(
            level=0).mean().add_suffix("_{}R".format(n_samples))
        return average_df

    def merge(self, results: pd.DataFrame, date: datetime.date, n_samples: int | str = "all") -> pd.DataFrame:
        df = results[results["date"] == date]
        average_df = self.average(date, n_samples)
        merge_df = pd.merge(
            df, average_df, left_on="racer_number", right_index=True, how="left")
        return merge_df

    def merge_all(self, results, n_samples: int | str = "all") -> pd.DataFrame:
        date_list = results["date"].unique()
        merge_all_df = pd.concat([self.merge(results, date, n_samples)
                                 for date in tqdm(date_list)])
        return merge_all_df

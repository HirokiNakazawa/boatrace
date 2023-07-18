import argparse
import pandas as pd
from datetime import datetime, timedelta

import utils

DEBUG = True


class BoatRace:
    def __init__(self, args):
        self.args = args

    def main(self):
        if self.args.scrape:
            # 年単位でスクレイピングする
            year = self.args.scrape
            program_list = utils.get_program_list(year)
            results = utils.get_scrape_results(program_list)
            infos = utils.get_scrape_infos(program_list)

            # データを保存
            utils.save_scrape_data(results, infos, year)

        elif self.args.update:
            # 当日までのデータをスクレイピングし、データを更新する
            results_all_latest = pd.read_pickle("data/results_all.pickle")
            results_r_latest = pd.read_pickle("data/results_r.pickle")
            returns_latest = pd.read_pickle("data/returns.pickle")

            latest_date = utils.get_latest_date(results_all_latest)
            today = datetime.now().date()
            yesterday = (today - timedelta(1))

            # 最新データが前日の場合と前日以前の場合で処理が分かれる
            if latest_date == today:
                print("データは最新の状態です")
            elif latest_date == yesterday:
                print("最新データは前日")
                program_list = utils.get_program_list(today=True)
                race_results = utils.get_scrape_results(program_list)
                race_infos = utils.get_scrape_infos(program_list)

                # データを更新
                utils.update_data(results_all_latest,
                                  results_r_latest, returns_latest, race_results, race_infos)
            else:
                print("最新データは前日以前")
                program_list = utils.get_between_program(latest_date)
                race_results = utils.get_scrape_results(program_list)
                race_infos = utils.get_scrape_infos(program_list)

                # データを更新
                utils.update_data(results_all_latest,
                                  race_results, race_infos, between=True)

        elif self.args.predict:
            # 当日の番組表をスクレイピングする
            results_all = pd.read_pickle("data/results_all.pickle")

            program_list = utils.get_program_list(today=True)
            race_infos = utils.get_scrape_infos(program_list)

            # 当日の予想を行う
            utils.predict(race_infos, results_all)

        elif self.args.check:
            pass
        elif self.args.model_update:
            pass
        elif self.args.debug:
            program_list = utils.get_program_list(yesterday=True)
            results = utils.get_scrape_results(program_list)
            infos = utils.get_scrape_infos(program_list)
            utils.save_scrape_data(results, infos)
        else:
            print(self.args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scrape", help="年を指定してスクレイピング")
    parser.add_argument(
        "-u", "--update", help="レース結果をスクレイピングし、データをアップデート", action="store_true")
    parser.add_argument("-p", "--predict", help="当日の予想",
                        action="store_true")
    parser.add_argument(
        "-c", "--check", help="現状のモデルの回収率を計算", action="store_true")
    parser.add_argument("-m", "--model_update", help="モデルをアップデート",
                        action="store_true")
    parser.add_argument("-d", "--debug", help="デバッグ用", action="store_true")
    args = parser.parse_args()

    boatrace = BoatRace(args)
    boatrace.main()

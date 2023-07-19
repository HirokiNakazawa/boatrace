import utils
import pandas as pd
import os
import argparse
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(
    filename="./logs/boatrace.log",
    level=logging.INFO,
    format=fmt,
)

rank = int(os.getenv("RANK"))
class_int = os.getenv("CLASS_INT")
number_del = os.getenv("NUMBER_DEL")
seed = int(os.getenv("SEED"))


class Log:
    @staticmethod
    def info(str):
        logger.info(str)


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

        elif self.args.model_create:
            # モデルを作成する
            results_all = utils.get_results_merge_infos()

            # race_idをindexに変換
            results_all.set_index("race_id", inplace=True)

            # カテゴリ変数化、前処理
            results_c = utils.process_categorical(
                results_all,
                rank=rank,
                class_int=class_int,
                number_del=number_del
            )

            # モデル作成
            lgb_train, lgb_valid, params = utils.get_lgb_train_valid(results_c)
            params_o = utils.get_optuna_params(
                params=params,
                lgb_train=lgb_train,
                lgb_valid=lgb_valid,
                seed=seed
            )
            X_train, y_train, X_test, y_test = utils.get_train_test(results_c)
            lgb_clf = utils.get_lgb_clf(params_o, X_train, y_train)

            # モデルを評価するためのreturnsを取得
            returns = utils.get_returns()

            # race_idをindexに変換
            returns.set_index("race_id", inplace=True)

            # モデルを評価
            gain_dict = utils.get_gain_dict(lgb_clf, returns, X_test)

            # 各掛け方の最大回収率を確認
            for k, v in gain_dict.items():
                print(k)
                print(v[v["n_bets"] > 200].sort_values(
                    "return_rate", ascending=False).head(3))

            # モデルとパラメータを保存
            utils.save_model(gain_dict, lgb_clf, rank,
                             class_int, number_del, seed)

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
            pass
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
    parser.add_argument("-mc", "--model_create", help="モデルを作成",
                        action="store_true")
    parser.add_argument("-mu", "--model_update", help="モデルをアップデート",
                        action="store_true")
    parser.add_argument("-d", "--debug", help="デバッグ用", action="store_true")
    args = parser.parse_args()

    boatrace = BoatRace(args)
    boatrace.main()

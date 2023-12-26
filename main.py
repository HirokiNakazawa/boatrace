from handle_db import HandleDB
from modules.model_evaluate import *
from modules.model_create import *
from modules.scrape_data import *
from modules.update_data import *
from modules.program_list import *
from modules.utils import *
from modules.predict import *
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


seed = int(os.getenv("SEED"))
threshold = float(os.getenv("THRESHOLD"))


class BoatRace:
    def __init__(self, args) -> None:
        self.args = args

    def main(self) -> None:
        if self.args.scrape:
            # 年単位でスクレイピングし、DBにデータを保存する
            year = self.args.scrape
            self.scrape(year)
        elif self.args.model_create:
            # モデルを作成する
            self.create_model()
        elif self.args.update:
            # 当日までのデータをスクレイピングし、データを更新する
            self.update()
        elif self.args.predict:
            # 当日のレースを予測する
            self.predict_today()
        elif self.args.check:
            # 現状のモデルの回収率を計算
            self.check_rate()
        elif self.args.save_data:
            # 現状のデータをpickleデータに変換し、保存
            self.save_data()
        elif self.args.debug:
            pass
        else:
            print(self.args)

    def scrape(self, year: str) -> None:
        """
        年単位でデータをスクレイピングし、DBにデータを格納する
        """
        program_list = get_program_list(year)
        results = get_scrape_results(program_list)
        infos = get_scrape_infos(program_list)
        returns = get_scrape_returns(program_list)

        # データを保存
        save_scrape_data(results, infos, returns, year)

    def create_model(self) -> None:
        """
        モデルを作成する
        """
        # DBからデータを取得する
        hdb = HandleDB()
        results_p = hdb.get_results()
        infos_p = hdb.get_infos()
        returns_p = hdb.get_returns()
        results_all = get_results_merge_infos(results_p, infos_p)

        # race_idをindexに変換
        results_all.set_index("race_id", inplace=True)
        returns_p.set_index("race_id", inplace=True)

        # カテゴリ変数化、前処理
        results_c = process_categorical(results_all)

        for seed in range(10, 31, 1):
            # モデル作成
            lgb_train, lgb_valid, params = get_lgb_train_valid(results_c)
            params_o = get_optuna_params(
                params=params,
                lgb_train=lgb_train,
                lgb_valid=lgb_valid,
                seed=seed
            )
            X_train, y_train, X_test, y_test = get_train_test(results_c)
            lgb_clf = get_lgb_clf(params_o, X_train, y_train)

            # モデルを評価し、gainの辞書を取得
            gain_dict = get_gain_dict(lgb_clf, returns_p, X_test)

            # gainの辞書を保存
            save_gain_dict(gain_dict, seed)

            # モデルを保存
            save_model(lgb_clf, seed)

    def update(self) -> None:
        """
        データを更新する
        """
        # DBからデータを取得する
        hdb = HandleDB()
        results_p = hdb.get_results()
        infos_p = hdb.get_infos()
        returns_p = hdb.get_returns()

        # race_idをindexに変換
        results_p.set_index("race_id", inplace=True)
        infos_p.set_index("race_id", inplace=True)
        returns_p.set_index("race_id", inplace=True)

        dt_now = datetime.now()
        year = dt_now.year

        update_results(results_p, year)
        update_infos(infos_p, year)
        update_returns(returns_p, year)

    def predict_today(self) -> None:
        """
        当日のレースを予測する
        """
        program_list = get_program_list(today=True)
        race_infos = get_scrape_infos(program_list)

        # 当日の予想を行う
        predict(race_infos, seed, threshold)

    def check_rate(self) -> None:
        """
        現状のモデルの回収率を算出する
        """
        # DBからデータを取得する
        hdb = HandleDB()
        results_db = hdb.get_results()
        infos_db = hdb.get_infos()
        returns_db = hdb.get_returns()
        results_all = get_results_merge_infos(results_db, infos_db)

        # race_idをindexに変換
        results_all.set_index("race_id", inplace=True)
        returns_db.set_index("race_id", inplace=True)

        # カテゴリ変数化、前処理
        results_c = process_categorical(results_all)

        X_train, y_train, X_test, y_test = get_train_test(results_c)

        # 回収率を算出
        check_model(returns_db, X_test, seed)

    def save_data(self) -> None:
        """
        現状のデータをpickleに変換し、保存する
        """
        # DBからデータを取得する
        hdb = HandleDB()
        results_db = hdb.get_results()
        infos_db = hdb.get_infos()
        returns_db = hdb.get_returns()

        # race_idをindexに変換
        results_db.set_index("race_id", inplace=True)
        infos_db.set_index("race_id", inplace=True)
        returns_db.set_index("race_id", inplace=True)

        # 保存
        results_db.to_dbickle("output/results_db.pickle")
        infos_db.to_dbickle("output/infos_db.pickle")
        returns_db.to_dbickle("output/returns_db.pickle")


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
    parser.add_argument("-sd", "--save_data",
                        help="DBのデータをpickleデータに変換し、保存", action="store_true")
    parser.add_argument("-d", "--debug", help="デバッグ用", action="store_true")
    args = parser.parse_args()

    boatrace = BoatRace(args)
    boatrace.main()

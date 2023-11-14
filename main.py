import utils
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()


rank = int(os.getenv("RANK"))
class_int = os.getenv("CLASS_INT", "False") == "True"
number_del = os.getenv("NUMBER_DEL", "False") == "True"
seed = int(os.getenv("SEED"))


class BoatRace:
    def __init__(self, args):
        self.args = args

    def main(self):
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
            # 当日の番組表をスクレイピングする
            self.predict_today()
        elif self.args.check:
            # 現状のモデルの回収率を計算
            self.check_rate()
        elif self.args.model_update:
            pass
        elif self.args.debug:
            program_list = utils.get_program_list_debug()
            results = utils.get_scrape_results(program_list)
            infos = utils.get_scrape_infos(program_list)

            # データを保存
            utils.save_scrape_data(results, infos)
        else:
            print(self.args)

    def scrape(self, year):
        """
        年単位でデータをスクレイピングし、DBにデータを格納する
        """
        program_list = utils.get_program_list(year)
        results = utils.get_scrape_results(program_list)
        infos = utils.get_scrape_infos(program_list)

        # データを保存
        utils.save_scrape_data(results, infos, year)

    def create_model(self):
        """
        モデルを作成する
        """
        # 最新のレース結果データと、最新の払い戻しデータが同じデータかを確認
        if utils.is_same_latest_data():
            pass
        else:
            print("レース結果と払い戻しの最新データに差異があるため、処理を中断します")
            return

        # DBからデータを取得する
        results_all = utils.get_results_merge_infos()
        returns = utils.get_returns()

        # race_idをindexに変換
        results_all.set_index("race_id", inplace=True)
        returns.set_index("race_id", inplace=True)

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

        # モデルを評価
        gain_dict = utils.get_gain_dict(lgb_clf, returns, X_test)

        # モデルとパラメータを保存
        utils.save_model(gain_dict, lgb_clf, rank,
                         class_int, number_del, seed)

    def update(self):
        """
        データを更新する
        """
        # 最新のレース結果データと、最新の払い戻しデータが同じデータかを確認
        if utils.is_same_latest_data():
            pass
        else:
            print("レース結果と払い戻しの最新データに差異があるため、処理を中断します")
            return

        # DBからデータを取得する
        results_all_latest = utils.get_results_merge_infos()

        latest_date = utils.get_latest_date(results_all_latest)
        to_date = latest_date + timedelta(1)
        dt_now = datetime.now()
        today = dt_now.date()
        yesterday = (today - timedelta(1))

        # 実行した時間帯により、最新データの対象が変わる
        if int(dt_now.strftime("%H")) < 23:
            if latest_date == yesterday:
                print("データは最新の状態です")
                return
            else:
                print("データは最新ではありません。更新が必要です")
                from_date = yesterday
                pass
        else:
            if latest_date == today:
                print("データは最新の状態です")
                return
            else:
                print("データは最新ではありません。更新が必要です")
                from_date = today
                pass

        # 過去データをスクレイピングする
        program_list = utils.get_between_program(to_date, from_date)
        print(program_list)

        results = utils.get_scrape_results(program_list)
        infos = utils.get_scrape_infos(program_list)

        # データを保存
        utils.save_scrape_data(results, infos)

    def predict_today(self):
        """
        当日のレースを予測する
        """
        program_list = utils.get_program_list(today=True)
        race_infos = utils.get_scrape_infos(program_list)

        # 当日の予想を行う
        utils.predict(race_infos)

    def check_rate(self):
        """
        現状のモデルの回収率を算出する
        """
        results_all = pd.read_pickle("data/results_all.pickle")
        returns = pd.read_pickle("data/returns.pickle")

        results_c = utils.process_categorical(results_all, 2, True, False)
        X_train, y_train, X_test, y_test = utils.get_train_test(results_c)

        # 回収率を算出
        utils.check_model(returns, X_test)


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

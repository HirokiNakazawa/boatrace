"""
データベース関連処理
"""

import sqlalchemy as sa
import pandas as pd
import os
import mysql.connector
import glob
import argparse
import logging
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(
    filename="./logs/handle_db.log",
    level=logging.INFO,
    format=fmt,
)

db_driver = os.getenv("DB_DRIVER")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

engine = sa.create_engine(
    sa.engine.url.URL.create(
        drivername=db_driver,
        username=db_user,
        password=db_pass,
        host=db_host,
        database=db_name,
    )
)


class Log:
    @staticmethod
    def info(str):
        logger.info(str)


class HandleDB:
    def __init__(self, args: str = None) -> None:
        self.args = args

    def main(self) -> None:
        if self.args.past:
            Log.info("過去データをデータベースに格納開始")
            self.insert_past_data()
        elif self.args.racer_results:
            Log.info("選手別データをデータベースに格納開始")
            self.insert_racer_results()
        else:
            print(self.args)

    def connect_db(self) -> None:
        """
        DB接続
        """
        self.conn = mysql.connector.connect(
            host=db_host, user=db_user, password=db_pass)
        Log.info("mysql connected")

    def close_db(self) -> None:
        """
        DB接続を切断
        """
        self.conn.close()
        Log.info("mysql close connected")

    def insert_past_data(self) -> None:
        """
        過去データをデータベースに格納する
        """
        dirs = ["infos", "results", "returns"]
        for dir in dirs:
            path = "res/%s/*.pickle" % dir
            files = glob.glob(path)
            for file in files:
                Log.info(file)
                df = pd.read_pickle(file)
                df.to_sql(dir, con=engine, if_exists="append", index=False)
        Log.info("過去データをデータベースに格納完了")

    def insert_scrape_data(self, results: pd.DataFrame, infos: pd.DataFrame, returns: pd.DataFrame) -> None:
        """
        スクレイピングしたデータをデータベースに格納する
        """
        Log.info("スクレイピングしたデータをデータベースに格納開始")
        results.to_sql("results", con=engine, if_exists="append", index=False)
        infos.to_sql("infos", con=engine, if_exists="append", index=False)
        returns.to_sql("returns", con=engine, if_exists="append", index=False)
        Log.info("スクレイピングしたデータをデータベースに格納完了")

    def insert_racer_results(self) -> None:
        """
        線種別データをデータベースに格納する
        """
        results_all = self.get_results_all()
        racer_results = results_all[[
            "race_id", "position", "boat_number", "racer_number", "date"]].copy()
        racer_results.to_sql("racer_results", con=engine,
                             if_exists="append", index=False)
        Log.info("選手別成績データをデータベースに格納完了")

    def get_results_latest(self) -> None:
        """
        DBのレース結果データの最新レースIDを返す
        """
        Log.info("resultsの最新レースIDを返す")
        sql = """
            SELECT
                race_id
            FROM
                results
            ORDER BY
                race_id DESC
            LIMIT 1;
            """
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
        return df["race_id"].values[0]

    def get_returns_latest(self) -> None:
        """
        DBの払い戻しデータの最新レースIDを返す
        """
        Log.info("returnsの最新レースIDを返す")
        sql = """
            SELECT
                race_id
            FROM
                returns
            ORDER BY
                race_id DESC
            LIMIT 1;
            """
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
        return df["race_id"].values[0]

    def get_results_all(self) -> None:
        """
        DBのデータを使用して、results_allを返す
        """
        Log.info("resultsとinfosを結合し、データフレーム形式で返す")
        sql = """
            SELECT
                r.race_id, r.position, r.boat_number, r.racer_number,
                i.age, i.weight, i.class, i.national_win_rate,
                i.national_second_rate, i.local_win_rate, i.local_second_rate,
                i.motor_second_rate, i.boat_second_rate, i.date
            FROM
                results as r
            JOIN
                infos as i
            ON
                r.race_id = i.race_id
            AND
                r.boat_number = i.boat_number
            AND
                r.racer_number = i.racer_number
            ORDER BY i.date DESC
            """
        df = pd.read_sql(
            sql=sql,
            con=engine,
        )
        return df

    def get_racer_results(self) -> None:
        """
        DBのデータを使用して、racer_resultsを返す
        """
        Log.info("racer_resultsをデータフレーム形式で返す")
        sql = """
            SELECT
                *
            FROM
                racer_results
            """
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
        return df

    def get_returns(self) -> None:
        """
        DBのデータを使用して、returnsを返す
        """
        Log.info("returnsをデータフレーム形式で返す")
        sql = """
            SELECT
                *
            FROM
                returns
            """
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--past", help="過去データをデータベースに格納", action="store_true")
    parser.add_argument("-rr", "--racer_results",
                        help="選手別成績データをデータベースに格納", action="store_true")
    args = parser.parse_args()

    handle_db = HandleDB(args)
    handle_db.main()

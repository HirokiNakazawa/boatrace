"""
データベース関連処理
"""

from modules.utils import *

import sqlalchemy as sa
import pandas as pd
import os
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
        else:
            print(self.args)

    def insert_past_data(self) -> None:
        """
        過去データをデータベースに格納する
        """
        self.insert_results_data(is_raw=True)
        self.insert_infos_data(is_raw=True)
        self.insert_returns_data(is_raw=True)
        Log.info("過去データをデータベースに格納完了")

    def insert_results_data(self, is_raw: bool = False, results_p: pd.DataFrame = None) -> None:
        """
        レース結果データをデータベースに格納する
        """
        if is_raw:
            path = "raw/results/*.pickle"
            files = glob.glob(path)
            for file in files:
                results = pd.read_pickle(file)
                df = get_results_p(results)
                df.to_sql("results", con=engine,
                          if_exists="append", index=True)
        else:
            results_p.to_sql("results", con=engine,
                             if_exists="append", index=True)

    def insert_infos_data(self, is_raw: bool = False, infos_p: pd.DataFrame = None) -> None:
        """
        レース情報データをデータベースに格納する
        """
        if is_raw:
            path = "raw/infos/*.pickle"
            files = glob.glob(path)
            for file in files:
                infos = pd.read_pickle(file)
                df = get_infos_p(infos)
                df.to_sql("infos", con=engine,
                          if_exists="append", index=True)
        else:
            infos_p.to_sql("infos", con=engine,
                           if_exists="append", index=True)

    def insert_returns_data(self, is_raw: bool = False, returns_p: pd.DataFrame = None) -> None:
        """
        払い戻し表データをデータベースに格納する
        """
        if is_raw:
            path = "raw/returns/*.pickle"
            files = glob.glob(path)
            for file in files:
                returns = pd.read_pickle(file)
                df = get_returns_p(returns)
                df.to_sql("returns", con=engine,
                          if_exists="append", index=True)
        else:
            returns_p.to_sql("returns", con=engine,
                             if_exists="append", index=True)

    def get_results(self) -> pd.DataFrame:
        """
        DBのレース結果データを返す
        """
        Log.info("resultsをデータフレーム形式で返す")
        sql = """
            SELECT
                *
            FROM
                results
            """
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
        return df

    def get_infos(self) -> pd.DataFrame:
        """
        DBのレース情報データを返す
        """
        Log.info("resultsをデータフレーム形式で返す")
        sql = """
            SELECT
                *
            FROM
                infos
            """
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
        return df

    def get_returns(self) -> pd.DataFrame:
        """
        DBの払い戻し表データを返す
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
    args = parser.parse_args()

    handle_db = HandleDB(args)
    handle_db.main()

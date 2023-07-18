"""
データベース関連処理
"""

import logging
import argparse
import glob
import mysql.connector
import pymysql
import pandas as pd
import sqlalchemy as sa

logger = logging.getLogger(__name__)

fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(
    filename="./logs/handle_db.log",
    level=logging.INFO,
    format=fmt,
)

db_user = "boatrace"
db_pass = "boatrace"
db_host = "127.0.0.1"
db_database = "boatrace_db"

engine = sa.create_engine(
    sa.engine.url.URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_pass,
        host=db_host,
        database=db_database,
    )
)


class Log:
    @staticmethod
    def info(str):
        logger.info(str)


class HandleDB:
    def __init__(self, args=None):
        self.args = args

    def connect_db(self):
        self.conn = mysql.connector.connect(
            host=db_host, user=db_user, password=db_pass)
        Log.info("mysql connected")

    def close_db(self):
        self.conn.close()
        Log.info("mysql close connected")

    def main(self):
        if self.args.past:
            Log.info("過去データをデータベースに格納開始")
            self.insert_past_data()

    def insert_past_data(self):
        dirs = ["infos", "results", "returns"]
        for dir in dirs:
            path = "res/%s/*.pickle" % dir
            files = glob.glob(path)
            for file in files:
                Log.info(file)
                df = pd.read_pickle(file)
                df.to_sql(dir, con=engine, if_exists="append", index=False)
        Log.info("過去データをデータベースに格納完了")

    def insert_scrape_data(self, results, infos, returns):
        Log.info("スクレイピングしたデータをデータベースに格納開始")
        results.to_sql("results", con=engine, if_exists="append", index=False)
        infos.to_sql("infos", con=engine, if_exists="append", index=False)
        returns.to_sql("returns", con=engine, if_exists="append", index=False)
        Log.info("スクレイピングしたデータをデータベースに格納完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--past", help="過去データをデータベースに格納", action="store_true")
    args = parser.parse_args()
    print(args)

    handle_db = HandleDB(args)
    handle_db.main()

import argparse
import pandas as pd
import mysql.connector
import logging

logger = logging.getLogger(__name__)

fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(
    filename="./logs/handle_db.log",
    level=logging.DEBUG,
    format=fmt,
)


class Log:
    @staticmethod
    def info(str):
        logger.info(str)


class HandleDB:
    def __init__(self, args):
        self.args = args

    def connect_db(self):
        self.conn = mysql.connector.connect(
            host='127.0.0.1', user='root', password='')
        Log.info("mysql connected")

    def close_db(self):
        self.conn.close()
        Log.info("mysql close connected")

    def main(self):
        self.connect_db()
        if self.args.past:
            Log.info("過去データをデータベースに格納開始")
            self.insert_past_data()
            pass
        self.close_db()

    def insert_past_data(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--past", help="過去データをデータベースに格納", action="store_true")
    args = parser.parse_args()

    handle_db = HandleDB(args)
    handle_db.main()

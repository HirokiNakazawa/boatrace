"""
プログラムリスト取得関連の関数群
"""

from datetime import datetime, timedelta
from typing import Tuple
import pandas as pd


def get_program_list(year: str = "2022", yesterday: bool = False, today: bool = False) -> list:
    """
    スクレイピングするプログラムリストを返す
    """
    program_list = []

    if yesterday or today:
        if yesterday:
            dt = datetime.now() - timedelta(1)
        if today:
            dt = datetime.now()
        year = dt.year
        month = dt.month
        day = dt.day
        for place in range(1, 25, 1):
            program_list.append(
                "%s%s/%s/%s" % (year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    else:
        for month in range(1, 13, 1):
            for place in range(1, 25, 1):
                for day in range(1, 32, 1):
                    program_list.append("%s%s/%s/%s" %
                                        (year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    return program_list


def get_latest_date(data: pd.DataFrame) -> datetime:
    df = data.copy()
    df["date"] = df.index.map(lambda x: datetime.strptime(
        f"{x[0:4]}-{x[4:6]}-{x[8:10]}", "%Y-%m-%d"))
    latest_date = df["date"].sort_values().tail(1).values[0]
    latest_date = datetime.fromtimestamp(
        latest_date.astype(datetime) * 1e-9).date()
    return latest_date


def get_between_from_to(latest_date: datetime) -> Tuple[datetime, datetime]:
    from_date = latest_date + timedelta(1)

    dt_now = datetime.now()
    today = dt_now.date()
    yesterday = (today - timedelta(1))

    # 実行した時間帯により、最新データの対象が変わる
    if int(dt_now.strftime("%H")) < 23:
        to_date = yesterday
    else:
        to_date = today

    return from_date, to_date


def get_between_program(from_date: datetime, to_date: datetime) -> list:
    """
    スクレイピング対象のプログラムリストを返す
    """
    from_split = from_date.strftime("%Y-%m-%d").split("-")
    from_year = from_split[0]
    from_month = from_split[1]
    from_day = from_split[2]

    to_split = to_date.strftime("%Y-%m-%d").split("-")
    to_year = to_split[0]
    to_month = to_split[1]
    to_day = to_split[2]

    program_list = []

    if int(from_month) == int(to_month):
        print("最新データは今月中")
        for place in range(1, 25, 1):
            for day in range(int(from_day), int(to_day) + 1, 1):
                program_list.append(
                    "%s%s/%s/%s" % (from_year, str(from_month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    elif int(from_month) != int(to_month):
        print("最新データは先月以前")
        for month in range(int(from_month), int(to_month) + 1, 1):
            for place in range(1, 25, 1):
                if int(to_month) == int(month):
                    for day in range(1, int(to_day) + 1, 1):
                        program_list.append(
                            "%s%s/%s/%s" % (from_year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
                else:
                    for day in range(1, 32, 1):
                        program_list.append(
                            "%s%s/%s/%s" % (from_year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
    return program_list

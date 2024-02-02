"""
データスクレイピング関連の関数群
"""

import pandas as pd

from modules.utils import *
from handle_db import HandleDB
from results import Results
from returns import Returns
from infos import Infos


def get_scrape_results(program_list: list) -> dict:
    """
    スクレイピングしたレース結果を返す
    """
    results = Results.scrape(program_list)
    return results


def get_scrape_infos(program_list: list) -> dict:
    """
    スクレイピングしたレース情報を返す
    """
    infos = Infos.scrape(program_list)
    return infos


def get_scrape_returns(program_list: list) -> dict:
    """
    スクレイピングした払い戻し表を返す
    """
    returns = Returns.scrape(program_list)
    return returns


def save_scrape_data(results: pd.DataFrame, infos: pd.DataFrame, returns: pd.DataFrame, year: str = "") -> None:
    """
    スクレイピングしたデータを保存する
    """
    results_p = get_results_p(results)
    infos_p = get_infos_p(infos)
    returns_p = get_returns_p(returns)

    if year:
        results_raw_name = f"raw/results/results_{year}.pickle"
        infos_raw_name = f"raw/infos/infos_{year}.pickle"
        returns_raw_name = f"raw/returns/returns_{year}.pickle"

        results_tmp_name = f"tmp/results/results_{year}.pickle"
        infos_tmp_name = f"tmp/infos/infos_{year}.pickle"
        returns_tmp_name = f"tmp/returns/returns_{year}.pickle"

        results.to_pickle(results_raw_name)
        infos.to_pickle(infos_raw_name)
        returns.to_pickle(returns_raw_name)

        results_p.to_pickle(results_tmp_name)
        infos_p.to_pickle(infos_tmp_name)
        returns_p.to_pickle(returns_tmp_name)

    hdb = HandleDB()
    hdb.insert_results_data(is_raw=False, results_p=results_p)
    hdb.insert_infos_data(is_raw=False, infos_p=infos_p)
    hdb.insert_returns_data(is_raw=False, returns_p=returns_p)

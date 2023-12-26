"""
データ更新関連の関数群
"""

import pandas as pd

from modules.program_list import *
from modules.scrape_data import *


def update_results(results_p: pd.DataFrame, year: int) -> None:
    """
    レース結果データを更新する
    """
    latest_date = get_latest_date(results_p)
    from_date, to_date = get_between_from_to(latest_date)
    if from_date > to_date:
        print("レース結果データは最新です")
        return
    else:
        print("レース結果データは最新ではありません。更新を開始します。")

    program_between_list = get_between_program(from_date, to_date)

    results_new = get_scrape_results(program_between_list)
    raw_file_path = Path(f"raw/results/results_{year}.pickle")
    save_update_data(results_new, raw_file_path)

    results_p_new = get_results_p(results_new)
    tmp_file_path = Path(f"tmp/results/results_{year}.pickle")
    save_update_data(results_p_new, tmp_file_path)

    hdb = HandleDB()
    hdb.insert_results_data(is_raw=False, results_p=results_p_new)


def update_infos(infos_p: pd.DataFrame, year: int) -> None:
    """
    レース情報データを更新する
    """
    latest_date = get_latest_date(infos_p)
    from_date, to_date = get_between_from_to(latest_date)
    if from_date > to_date:
        print("レース情報データは最新です")
        return
    else:
        print("レース情報データは最新ではありません。更新を開始します。")

    program_between_list = get_between_program(from_date, to_date)

    infos_new = get_scrape_infos(program_between_list)
    raw_file_path = Path(f"raw/infos/infos_{year}.pickle")
    save_update_data(infos_new, raw_file_path)

    infos_p_new = get_infos_p(infos_new)
    tmp_file_path = Path(f"tmp/infos/infos_{year}.pickle")
    save_update_data(infos_p_new, tmp_file_path)

    hdb = HandleDB()
    hdb.insert_infos_data(is_raw=False, infos_p=infos_p_new)


def update_returns(returns_p: pd.DataFrame, year: int) -> None:
    """
    払い戻し表データを更新する
    """
    latest_date = get_latest_date(returns_p)
    from_date, to_date = get_between_from_to(latest_date)
    if from_date > to_date:
        print("払い戻し表データは最新です")
        return
    else:
        print("払い戻し表データは最新ではありません。更新を開始します。")

    program_between_list = get_between_program(from_date, to_date)

    returns_new = get_scrape_returns(program_between_list)
    raw_file_path = Path(f"raw/returns/returns_{year}.pickle")
    save_update_data(returns_new, raw_file_path)

    returns_p_new = get_returns_p(returns_new)
    tmp_file_path = Path(f"tmp/returns/returns_{year}.pickle")
    save_update_data(returns_p_new, tmp_file_path)

    hdb = HandleDB()
    hdb.insert_returns_data(is_raw=False, returns_p=returns_p_new)

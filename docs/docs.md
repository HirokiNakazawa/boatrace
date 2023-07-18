# 競艇予想AI

## コマンドライン引数
- -s, --scrape YYYY  
  渡された年のデータをスクレイピングし、ファイル保存
- -u, --update  
  データをアップデートする
- -p, --predict  
  当日の予想を行う
- -c, --check  
  現状のモデルの回収率を計算
- -m. --model_update  
  モデルをアップデートする

## データの種類と前処理、保存の流れ
- 期間を指定してスクレイピングした生データ・・・json保存できる形にする必要あり
  - race_results_YYYY(json)  
  - race_infos_YYYY(json)
- 生データに前処理をかけたデータ
  - results_YYYY(dataframe)・・・DB保存
  - infos_YYYY(dataframe)・・・DB保存
  - returns_YYYY(dataframe)・・・race_results_YYYYを使用して作成する。DB保存
- 前処理をかけた結果と情報を結合したデータ
  - results_all_YYYY(dataframe)
- 全期間結合したモデル作成の基本データ
  - results_all(dataframe)
- 着順の平均を集計して加えたデータ
  - results_r(dataframe)

## データの更新

### 更新タイミング
レース前日の23時

### 更新方法
- 最新データが前日の場合  
  当日のデータをスクレイピングした後、results_allとresults_rを更新
- 最新データが前日よりも前の場合  
  当日までのプログラムリストを作成し、スクレイピング  
  1日ずつresults_allとresults_rを更新  

## データベース
データ保存をデータベースで行う  
- DB名  
  boatrace_db
- テーブル
  - results
    - race_id(char(20))
    - position(int)
    - boat_number(char(1))
    - racer_number(char(4))
    - start_time(char(10))
    - race_time(char(10))
  - infos
    - race_id(char(20))
    - boat_number(char(1))
    - racer_number(char(4))
    - age(int)
    - weight(int)
    - class(char(2))
    - national_win_rate(float)
    - national_second_rate(float)
    - local_win_rate(float)
    - local_second_rate(float)
    - date(datetime)
  - returns
    - race_id(char(20))
    - win_t(int)
    - return_t(int)
    - win_f1(int)
    - return_f1(int)
    - win_f2(int)
    - return_f2(int)
    - win_2t_1(int)
    - win_2t_2(int)
    - return_2t(int)
    - win_2f_1(int)
    - win_2f_2(int)
    - return_2f(int)
    - win_3t_1(int)
    - win_3t_2(int)
    - win_3t_3(int)
    - return_3t(int)
    - win_3f_1(int)
    - win_3f_2(int)
    - win_3f_3(int)
    - return_3f(int)

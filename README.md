# 競艇予想プログラム

## コマンドライン引数
- -s, --scrape YYYY  
  渡された年のデータをスクレイピングし、ファイル保存
- -u, --update  
  データをアップデートする
- -p, --predict  
  当日の予想を行う
- -c, --check  
  現状のモデルの回収率を計算
- -mc. --model_create  
  モデルを作成する
- -sd, --save_data  
  現状のデータをpickleデータで保存する
- -d, --debug  
  デバッグ用

## データの種類と前処理、保存の流れ
- 期間を指定してスクレイピングした生データ  
  | 変数名       | 保存先                          |
  | ------------ | ------------------------------- |
  | results_YYYY | raw/results/results_YYYY.pickle |
  | infos_YYYY   | raw/infos/infos_YYYY.pickle     |
  | returns_YYYY | raw/returns/returns_YYYY.pickle |
- 整形済みデータ
  | 変数名         | 保存先                          |
  | -------------- | ------------------------------- |
  | results_p_YYYY | tmp/results/results_YYYY.pickle |
  | infos_p_YYYY   | tmp/infos/infos_YYYY.pickle     |
  | returns_p_YYYY | tmp/returns/returns_YYYY.pickle |
- -sdオプションによって出力されるデータ
  | 変数名     | 出力先                |
  | ---------- | --------------------- |
  | results_db | output/results.pickle |
  | infos_db   | output/infos.pickle   |
  | returns_db | output/returns.pickle |

## データの更新

### 更新タイミング
任意のタイミング(実行タイミングによって動的に変数の内容を変更)

### 更新方法
- 最新データが前日よりも前の場合  
  当日までのプログラムリストを作成し、スクレイピング  

## データベース
データ保存をデータベースで行う

### DB  
boatrace_db

### テーブル
- results
  | カラム名     | 型定義      |
  | ------------ | ----------- |
  | race_id      | varchar(20) |
  | position     | int         |
  | boat_number  | varchar(1)  |
  | racer_number | varchar(4)  |
- infos
  | カラム名             | 型定義      |
  | -------------------- | ----------- |
  | race_id              | varchar(20) |
  | boat_number          | varchar(1)  |
  | date                 | datetime    |
  | racer_number         | varchar(4)  |
  | class                | varchar(2)  |
  | age                  | int         |
  | national_win_rate    | float       |
  | national_second_rate | float       |
  | national_third_rate  | float       |
  | local_win_rate       | float       |
  | local_second_rate    | float       |
  | local_third_rate     | float       |
  | motor_second_rate    | float       |
  | motor_third_rate     | float       |
  | boat_second_rate     | float       |
  | boat_third_rate      | float       |
- returns
  | カラム名  | 型定義      |
  | --------- | ----------- |
  | race_id   | varchar(20) |
  | win_t     | int         |
  | return_t  | int         |
  | win_f1    | int         |
  | return_f1 | int         |
  | win_f2    | int         |
  | return_f2 | int         |
  | win_2_1   | int         |
  | win_2_2   | int         |
  | return_2t | int         |
  | return_2f | int         |
  | win_3_1   | int         |
  | win_3_2   | int         |
  | win_3_3   | int         |
  | return_3t | int         |
  | return_3f | int         |

## 各パラメータファイル
- gain_*.json
  モデルの閾値別の回収率ファイル
- model_*.pickle  
  モデルを格納したファイル
- thresholds.json  
  予想の際に使用する閾値ファイル
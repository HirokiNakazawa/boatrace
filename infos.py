from tqdm import tqdm
import pandas as pd
import time
import re
import unicodedata


class Infos:
    def __init__(self, infos: pd.DataFrame) -> None:
        self.infos = infos

    @staticmethod
    def scrape(program_list: list) -> pd.DataFrame:
        race_infos = {}
        for program in tqdm(program_list):
            try:
                time.sleep(1)
                program_split = program.split("/")
                year = program_split[0][0:4]
                month = program_split[0][4:6]
                day = program_split[2]
                hd = program_split[0] + day
                jcd = program_split[1]
                for i, rno in enumerate(range(1, 13, 1)):
                    try:
                        race_id = program.replace("/", "") + str(rno).zfill(2)
                        url = "https://www.boatrace.jp/owpc/pc/race/racelist?rno=%s&jcd=%s&hd=%s" % (
                            str(rno).zfill(2), jcd, hd)
                        df_row = pd.read_html(url, flavor='lxml')

                        date_df = df_row[0]
                        date_df.drop(["レース", "レース.1"], axis=1, inplace=True)
                        date_list = list(date_df.values)[0]
                        date = pd.to_datetime(
                            "%s-%s-%s %s" % (year, month, day, date_list[i]))

                        infos_df = df_row[1]
                        df_droplevel1 = infos_df.copy()
                        df_droplevel1.columns = infos_df.columns.droplevel(1)
                        df_flat = df_droplevel1.copy()
                        df_flat.columns = ['_'.join(col).replace(
                            " ", "_") for col in df_droplevel1.columns.values]

                        df = df_flat[["枠_枠", "ボートレーサー_登録番号/級別_氏名_支部/出身地_年齢/体重", "ボートレーサー_F数_L数_平均ST",
                                      "全国_勝率_2連率_3連率", "当地_勝率_2連率_3連率", "モーター_No_2連率_3連率", "ボート_No_2連率_3連率"]].copy()
                        df.drop_duplicates(inplace=True)
                        df.rename(columns={"枠_枠": "boat_number", "ボートレーサー_登録番号/級別_氏名_支部/出身地_年齢/体重": "infos_main",
                                           "ボートレーサー_F数_L数_平均ST": "infos_sub", "全国_勝率_2連率_3連率": "national",
                                           "当地_勝率_2連率_3連率": "local", "モーター_No_2連率_3連率": "motor", "ボート_No_2連率_3連率": "boat"}, inplace=True)

                        df["date"] = [date] * len(df)
                        df.index = [race_id] * len(df)
                        race_infos[race_id] = df

                    except ValueError:
                        continue
                    except Exception as e:
                        print(e)
                        break
                    except:
                        break

            except ValueError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        race_infos_df = pd.concat([race_infos[key] for key in race_infos])
        return race_infos_df

    def preprocessing(self) -> None:
        df = self.infos.copy()

        df["racer_number"] = df["infos_main"].map(
            lambda x: re.findall(r"\d+", x)[0])
        df["class"] = df["infos_main"].map(
            lambda x: re.findall(r"[A-Z]\d", x)[0])
        df["age"] = df["infos_main"].map(lambda x: re.findall(r"\d+", x)[2])
        df["national_win_rate"] = df["national"].map(
            lambda x: re.findall(r"\S+", x)[0])
        df["national_second_rate"] = df["national"].map(
            lambda x: re.findall(r"\S+", x)[1])
        df["national_third_rate"] = df["national"].map(
            lambda x: re.findall(r"\S+", x)[2])
        df["local_win_rate"] = df["local"].map(
            lambda x: re.findall(r"\S+", x)[0])
        df["local_second_rate"] = df["local"].map(
            lambda x: re.findall(r"\S+", x)[1])
        df["local_third_rate"] = df["local"].map(
            lambda x: re.findall(r"\S+", x)[2])
        df["motor_second_rate"] = df["motor"].map(
            lambda x: re.findall(r"\S+", x)[1])
        df["motor_third_rate"] = df["motor"].map(
            lambda x: re.findall(r"\S+", x)[2])
        df["boat_second_rate"] = df["boat"].map(
            lambda x: re.findall(r"\S+", x)[1])
        df["boat_third_rate"] = df["boat"].map(
            lambda x: re.findall(r"\S+", x)[2])

        df["boat_number"] = df["boat_number"].map(
            lambda x: unicodedata.normalize("NFKC", x))
        df["boat_number"] = df["boat_number"].astype(int)
        df["racer_number"] = df["racer_number"].astype(int)
        df["age"] = df["age"].astype(int)
        df["national_win_rate"] = df["national_win_rate"].astype(float)
        df["national_second_rate"] = df["national_second_rate"].astype(float)
        df["national_third_rate"] = df["national_third_rate"].astype(float)
        df["local_win_rate"] = df["local_win_rate"].astype(float)
        df["local_second_rate"] = df["local_second_rate"].astype(float)
        df["local_third_rate"] = df["local_third_rate"].astype(float)
        df["motor_second_rate"] = df["motor_second_rate"].astype(float)
        df["motor_third_rate"] = df["motor_third_rate"].astype(float)
        df["boat_second_rate"] = df["boat_second_rate"].astype(float)
        df["boat_third_rate"] = df["boat_third_rate"].astype(float)

        df.drop(["infos_main", "infos_sub", "national",
                "local", "motor", "boat"], axis=1, inplace=True)
        df.index.name = "race_id"
        self.infos_p = df

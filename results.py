from tqdm import tqdm
import pandas as pd
import time
import re
import unicodedata


class Results:
    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results

    @staticmethod
    def scrape(program_list: list) -> pd.DataFrame:
        race_results = {}
        for program in tqdm(program_list):
            try:
                time.sleep(1)
                program_split = program.split("/")
                day = program_split[2]
                hd = program_split[0] + day
                jcd = program_split[1]
                for i, rno in enumerate(range(1, 13, 1)):
                    try:
                        race_id = program.replace("/", "") + str(rno).zfill(2)
                        url = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno=%s&jcd=%s&hd=%s" % (
                            str(rno).zfill(2), jcd, hd)
                        df = pd.read_html(url, flavor="lxml")[1]

                        df.index = [race_id] * len(df)
                        race_results[race_id] = df

                    except ValueError:
                        continue
                    except IndexError:
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

        race_results_df = pd.concat([race_results[key]
                                    for key in race_results])
        return race_results_df

    def preprocessing(self) -> None:
        df = self.results.copy()

        df["position"] = df["着"].map(
            lambda x: unicodedata.normalize("NFKC", x))
        df["racer_number"] = df["ボートレーサー"].map(
            lambda x: re.findall(r"\d+", x)[0])

        df.rename(columns={"枠": "boat_number"}, inplace=True)
        df.drop(["着", "ボートレーサー", "レースタイム"], axis=1, inplace=True)

        numeric_rows = pd.to_numeric(df["position"], errors="coerce").isnull()
        df = df[~df.index.isin(df[numeric_rows].index)]

        df["position"] = df["position"].astype(int)
        df["boat_number"] = df["boat_number"].astype(int)
        df["racer_number"] = df["racer_number"].astype(int)

        df.index.name = "race_id"
        self.results_p = df

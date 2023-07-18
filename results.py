from tqdm import tqdm
import requests
from requests import Response
from bs4 import BeautifulSoup
import time
import re
import pandas as pd


class Results:
    def __init__(self, race_results):
        self.results = race_results

    @staticmethod
    def scrape(program_list: list):
        def get_pre_list(url, response):
            soup: BeautifulSoup = BeautifulSoup(response.content, "lxml")
            pre_list = soup.find_all("pre")
            return pre_list

        results = {}
        for program in tqdm(program_list):
            time.sleep(1)
            try:
                url = "https://www1.mbrace.or.jp/od2/K/%s.html" % program
                response: Response = requests.get(url)
                if response.status_code == 404:
                    continue
                results[program] = get_pre_list(url, response)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break
        return results

    def preprocessing(self):
        results = self.results.copy()

        results_list = []
        for key in tqdm(list(results.keys())):
            for datas in results[key][1:]:
                datas = datas.replace("\u3000", "").split("\n")
                race_id = key.replace(
                    "/", "") + (re.search(r"\d+R", datas[0]).group()[:-1]).zfill(2)
                data_dict = {}
                data_dict["race_id"] = []
                data_dict["着順"] = []
                data_dict["艇番"] = []
                data_dict["選手番号"] = []
                data_dict["start_time"] = []
                data_dict["race_time"] = []
                for data in datas:
                    data_p = re.findall(r"\S+", data)
                    if re.match("^\s+0[1-6]", data):
                        data_dict["race_id"].append(race_id)
                        data_dict["着順"].append(int(data_p[0][-1]))
                        data_dict["艇番"].append(data_p[1])
                        data_dict["選手番号"].append(data_p[2])
                        data_dict["start_time"].append(data_p[8])
                        data_dict["race_time"].append(data_p[9])

                if len(data_dict["着順"]) == 6:
                    df = pd.DataFrame(data_dict)
                    results_list.append(df.copy())
                else:
                    continue

        self.results_p = pd.concat(results_list)
        self.results_p.set_index("race_id", inplace=True)

    def merge_infos(self, results, infos):
        results_merge_infos = pd.merge(
            results, infos, on=["race_id", "艇番", "選手番号"], how="left")
        return results_merge_infos

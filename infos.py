from tqdm import tqdm
import requests
from requests import Response
from bs4 import BeautifulSoup
import time
import re
import pandas as pd
from datetime import datetime


class Infos:
    def __init__(self, race_infos):
        self.infos = race_infos

    @staticmethod
    def scrape(program_list: list):
        def get_pre_list(url, response):
            soup: BeautifulSoup = BeautifulSoup(response.content, "lxml")
            pre_list = soup.find_all("pre")
            return pre_list

        infos = {}
        for program in tqdm(program_list):
            time.sleep(1)
            try:
                url = "https://www1.mbrace.or.jp/od2/B/%s.html" % program
                response: Response = requests.get(url)
                if response.status_code == 404:
                    continue
                infos[program] = get_pre_list(url, response)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break
        return infos

    def preprocessing(self):
        infos = self.infos.copy()

        infos_list = []
        for key in tqdm(list(infos.keys())):
            for datas in infos[key][1:]:
                datas = datas.replace("\u3000", "").split("\n")
                race_id = key.replace(
                    "/", "") + (re.search(r"\d+R", datas[0]).group()[:-1]).zfill(2)
                date_list = key.split("/")
                time = re.search(r"\d+:\d+", datas[0]).group()
                date_str = (
                    "%s-%s-%s %s" % (date_list[0][:4], date_list[0][4:], date_list[2], time))
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')

                data_dict = {}
                data_dict["race_id"] = []
                data_dict["艇番"] = []
                data_dict["選手番号"] = []
                data_dict["age"] = []
                data_dict["weight"] = []
                data_dict["class"] = []
                data_dict["全国勝率"] = []
                data_dict["全国2率"] = []
                data_dict["当地勝率"] = []
                data_dict["当地2率"] = []
                data_dict["date"] = []
                for data in datas:
                    if re.match(r"[0-9]\s\d+", data):
                        if "10.00" in data:
                            data = data.replace("10.00", " 9.99")
                        if "100.00" in data:
                            data = data.replace("100.00", " 9.99")
                        if "新規" in data:
                            data = data.replace("新規", " 0.00")
                        if re.search(r"\d+\.\d+\.", data):
                            data = re.sub(r"(\d{2}\.)", r" \1", data)
                        data_p = re.findall(r"\S+", data)
                        data_dict["race_id"].append(race_id)
                        data_dict["date"].append(date)
                        data_dict["艇番"].append(data_p[0])
                        data_dict["選手番号"].append(
                            re.match(r"^\d+", data_p[1]).group())
                        data_dict["age"].append(
                            int(re.match(r"(^\d+\D+)(\d+)", data_p[1]).group(2)))
                        data_dict["weight"].append(
                            int(re.match(r"(^\d+\D+\d+\D+)(\d+)", data_p[1]).group(2)))
                        data_dict["class"].append(
                            re.match(r"(^\d+\D+\d+\D+\d+)(.*)", data_p[1]).group(2))
                        data_dict["全国勝率"].append(float(data_p[2]))
                        data_dict["全国2率"].append(float(data_p[3]))
                        data_dict["当地勝率"].append(float(data_p[4]))
                        data_dict["当地2率"].append(float(data_p[5]))
                df = pd.DataFrame(data_dict)
                infos_list.append(df.copy())

        self.infos_p = pd.concat(infos_list)
        self.infos_p.set_index("race_id", inplace=True)
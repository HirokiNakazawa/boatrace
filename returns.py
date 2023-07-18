from tqdm.notebook import tqdm
import pandas as pd
import re


class Returns:
    def __init__(self, race_results):
        self.results = race_results

    def preprocessing(self):
        results = self.results.copy()
        return_list = []
        for key in tqdm(list(results.keys())):
            for datas in results[key][1:]:
                try:
                    datas = datas.replace("\u3000", "").split("\n")

                    data_dict = {}
                    data_dict["race_id"] = []
                    data_dict["win_t"] = []
                    data_dict["return_t"] = []
                    data_dict["win_f1"] = []
                    data_dict["return_f1"] = []
                    data_dict["win_f2"] = []
                    data_dict["return_f2"] = []
                    data_dict["win_2t_1"] = []
                    data_dict["win_2t_2"] = []
                    data_dict["return_2t"] = []
                    data_dict["win_2f_1"] = []
                    data_dict["win_2f_2"] = []
                    data_dict["return_2f"] = []
                    data_dict["win_3t_1"] = []
                    data_dict["win_3t_2"] = []
                    data_dict["win_3t_3"] = []
                    data_dict["return_3t"] = []
                    data_dict["win_3f_1"] = []
                    data_dict["win_3f_2"] = []
                    data_dict["win_3f_3"] = []
                    data_dict["return_3f"] = []

                    for data in datas:
                        data_p = re.findall(r"\S+", data)
                        if re.search("[0-9]R", data):
                            data_dict["race_id"].append(key.replace(
                                "/", "") + (re.search(r"\d+R", data).group()[:-1]).zfill(2))
                        if "単勝" in data:
                            data_dict["win_t"].append(int(data_p[1]))
                            data_dict["return_t"].append(int(data_p[2]))
                        if "複勝" in data:
                            data_dict["win_f1"].append(int(data_p[1]))
                            data_dict["return_f1"].append(int(data_p[2]))
                            data_dict["win_f2"].append(int(data_p[3]))
                            data_dict["return_f2"].append(int(data_p[4]))
                        if "２連単" in data:
                            data_dict["win_2t_1"].append(
                                int(data_p[1].split("-")[0]))
                            data_dict["win_2t_2"].append(
                                int(data_p[1].split("-")[1]))
                            data_dict["return_2t"].append(int(data_p[2]))
                        if "２連複" in data:
                            data_dict["win_2f_1"].append(
                                int(data_p[1].split("-")[0]))
                            data_dict["win_2f_2"].append(
                                int(data_p[1].split("-")[1]))
                            data_dict["return_2f"].append(int(data_p[2]))
                        if "３連単" in data:
                            data_dict["win_3t_1"].append(
                                int(data_p[1].split("-")[0]))
                            data_dict["win_3t_2"].append(
                                int(data_p[1].split("-")[1]))
                            data_dict["win_3t_3"].append(
                                int(data_p[1].split("-")[2]))
                            data_dict["return_3t"].append(int(data_p[2]))
                        if "３連複" in data:
                            data_dict["win_3f_1"].append(
                                int(data_p[1].split("-")[0]))
                            data_dict["win_3f_2"].append(
                                int(data_p[1].split("-")[1]))
                            data_dict["win_3f_3"].append(
                                int(data_p[1].split("-")[2]))
                            data_dict["return_3f"].append(int(data_p[2]))
                    df = pd.DataFrame(data_dict)
                    return_list.append(df.copy())
                except:
                    break

        self.returns_p = pd.concat(return_list)

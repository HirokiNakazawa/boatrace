from tqdm import tqdm
import pandas as pd
import time


class Returns:
    def __init__(self, returns: pd.DataFrame) -> None:
        self.returns = returns

    @staticmethod
    def scrape(program_list: list) -> pd.DataFrame:
        returns = {}
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
                        df = pd.read_html(url, flavor="lxml")[3]

                        drop_target = df["組番"].isnull()
                        df = df[~drop_target]

                        df.index = [race_id] * len(df)
                        returns[race_id] = df

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

        returns_df = pd.concat([returns[key] for key in returns])
        return returns_df

    def preprocessing(self) -> None:
        data = self.returns.copy()
        race_id_list = data.index.unique()

        returns_all_dict = {}
        for race_id in tqdm(race_id_list):
            try:
                returns_dict = {}
                df = data[data.index == race_id]
                df = df.replace(["¥", ","], "", regex=True)
                returns_dict["win_t"] = int(df[df["勝式"] == "単勝"]["組番"].values)
                returns_dict["return_t"] = int(
                    df[df["勝式"] == "単勝"]["払戻金"].values)
                returns_dict["win_f1"] = int(
                    df[df["勝式"] == "複勝"]["組番"].values[0])
                returns_dict["return_f1"] = int(
                    df[df["勝式"] == "複勝"]["払戻金"].values[0])
                returns_dict["win_f2"] = int(
                    df[df["勝式"] == "複勝"]["組番"].values[1])
                returns_dict["return_f2"] = int(
                    df[df["勝式"] == "複勝"]["払戻金"].values[1])
                returns_dict["win_2_1"] = int(
                    df[df["勝式"] == "2連単"]["組番"].values[0].split("-")[0])
                returns_dict["win_2_2"] = int(
                    df[df["勝式"] == "2連単"]["組番"].values[0].split("-")[1])
                returns_dict["return_2t"] = int(
                    df[df["勝式"] == "2連単"]["払戻金"].values[0])
                returns_dict["return_2f"] = int(
                    df[df["勝式"] == "2連複"]["払戻金"].values[0])
                returns_dict["win_3_1"] = int(
                    df[df["勝式"] == "3連単"]["組番"].values[0].split("-")[0])
                returns_dict["win_3_2"] = int(
                    df[df["勝式"] == "3連単"]["組番"].values[0].split("-")[1])
                returns_dict["win_3_3"] = int(
                    df[df["勝式"] == "3連単"]["組番"].values[0].split("-")[2])
                returns_dict["return_3t"] = int(
                    df[df["勝式"] == "3連単"]["払戻金"].values[0])
                returns_dict["return_3f"] = int(
                    df[df["勝式"] == "3連複"]["払戻金"].values[0])
                returns_all_dict[race_id] = returns_dict
            except IndexError:
                continue
            except ValueError:
                continue
            except TypeError:
                continue
            except Exception as e:
                print(e)
                print(race_id)
                break
            except:
                break

        df = pd.DataFrame.from_dict(returns_all_dict, orient="index")
        df.fillna(0, inplace=True)
        df.index.name = "race_id"
        self.returns_p = df

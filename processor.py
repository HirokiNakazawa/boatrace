class DataProcessor:
    def __init__(self):
        pass

    def get_program_list(year: str = "2022"):
        program_list = []
        for month in range(1, 13, 1):
            for place in range(1, 25, 1):
                for day in range(1, 32, 1):
                    program_list.append(
                        "%s%s/%s/%s" % (year, str(month).zfill(2), str(place).zfill(2), str(day).zfill(2)))
        return program_list

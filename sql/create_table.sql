USE boatrace_db;

DROP TABLE IF EXISTS results;
DROP TABLE IF EXISTS infos;
DROP TABLE IF EXISTS returns;

CREATE TABLE results(
    race_id varchar(20) NOT NULL,
    position int NOT NULL,
    boat_number varchar(1) NOT NULL,
    racer_number varchar(4) NOT NULL,
    start_time varchar(10) NOT NULL,
    race_time varchar(10) NOT NULL
);

CREATE TABLE infos(
    race_id varchar(20) NOT NULL,
    boat_number varchar(1) NOT NULL,
    racer_number varchar(4) NOT NULL,
    age int NOT NULL,
    weight int NOT NULL,
    class varchar(2) NOT NULL,
    national_win_rate float NOT NULL,
    national_second_rate float NOT NULL,
    local_win_rate float NOT NULL,
    local_second_rate float NOT NULL,
    date datetime NOT NULL
);

CREATE TABLE returns(
    race_id varchar(20) NOT NULL,
    win_t int NOT NULL,
    return_t int NOT NULL,
    win_f1 int NOT NULL,
    return_f1 int NOT NULL,
    win_f2 int NOT NULL,
    return_f2 int NOT NULL,
    win_2t_1 int NOT NULL,
    win_2t_2 int NOT NULL,
    return_2t int NOT NULL,
    win_2f_1 int NOT NULL,
    win_2f_2 int NOT NULL,
    return_2f int NOT NULL,
    win_3t_1 int NOT NULL,
    win_3t_2 int NOT NULL,
    win_3t_3 int NOT NULL,
    return_3t int NOT NULL,
    win_3f_1 int NOT NULL,
    win_3f_2 int NOT NULL,
    win_3f_3 int NOT NULL,
    return_3f int NOT NULL
);

USE boatrace_db;

DROP TABLE IF EXISTS results;
DROP TABLE IF EXISTS infos;
DROP TABLE IF EXISTS returns;

CREATE TABLE results(
    race_id varchar(20) NOT NULL,
    position int NOT NULL,
    boat_number varchar(1) NOT NULL,
    racer_number varchar(4) NOT NULL
);

CREATE TABLE infos(
    race_id varchar(20) NOT NULL,
    boat_number varchar(1) NOT NULL,
    date datetime NOT NULL,
    racer_number varchar(4) NOT NULL,
    class varchar(2) NOT NULL,
    age int NOT NULL,
    national_win_rate float NOT NULL,
    national_second_rate float NOT NULL,
    national_third_rate float NOT NULL,
    local_win_rate float NOT NULL,
    local_second_rate float NOT NULL,
    local_third_rate float NOT NULL,
    motor_second_rate float NOT NULL,
    motor_third_rate float NOT NULL,
    boat_second_rate float NOT NULL,
    boat_third_rate float NOT NULL
);

CREATE TABLE returns(
    race_id varchar(20) NOT NULL,
    win_t int NOT NULL,
    return_t int NOT NULL,
    win_f1 int NOT NULL,
    return_f1 int NOT NULL,
    win_f2 int NOT NULL,
    return_f2 int NOT NULL,
    win_2_1 int NOT NULL,
    win_2_2 int NOT NULL,
    return_2t int NOT NULL,
    return_2f int NOT NULL,
    win_3_1 int NOT NULL,
    win_3_2 int NOT NULL,
    win_3_3 int NOT NULL,
    return_3t int NOT NULL,
    return_3f int NOT NULL
);

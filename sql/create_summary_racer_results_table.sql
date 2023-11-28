use boatrace_db;

DROP TABLE IF EXISTS summary_racer_results;

CREATE TABLE summary_racer_results(
    racer_number varchar(4) NOT NULL,
    race_id varchar(20) NOT NULL,
    position_1R float,
    position_2R float,
    position_3R float,
    position_5R float,
    position_9R float,
    position_allR float
)
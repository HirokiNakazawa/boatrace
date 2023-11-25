use boatrace_db;

DROP TABLE IF EXISTS racer_results;

CREATE TABLE racer_results(
    racer_number varchar(4) NOT NULL,
    race_id varchar(20) NOT NULL,
    position int NOT NULL,
    boat_number varchar(1) NOT NULL,
    date datetime NOT NULL
);

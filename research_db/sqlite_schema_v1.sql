-- KEIBALAB Research DB schema v2
-- SQLite-first, portable to PostgreSQL. Research branch only.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL,
  description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_batches (
  batch_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('running','completed','partial','failed')),
  rows_read INTEGER NOT NULL DEFAULT 0 CHECK (rows_read >= 0),
  rows_inserted INTEGER NOT NULL DEFAULT 0 CHECK (rows_inserted >= 0),
  rows_rejected INTEGER NOT NULL DEFAULT 0 CHECK (rows_rejected >= 0),
  checksum TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS races (
  race_id TEXT PRIMARY KEY,
  race_date TEXT NOT NULL,
  venue TEXT NOT NULL,
  race_type TEXT NOT NULL DEFAULT 'flat' CHECK (race_type IN ('flat','banei','unknown')),
  race_number INTEGER NOT NULL CHECK (race_number > 0),
  race_name TEXT,
  surface TEXT,
  distance_m INTEGER CHECK (distance_m IS NULL OR distance_m > 0),
  going TEXT,
  scheduled_post_at TEXT,
  field_size INTEGER CHECK (field_size IS NULL OR field_size >= 0),
  source TEXT NOT NULL,
  source_updated_at TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  UNIQUE (race_date, venue, race_number)
);

CREATE TABLE IF NOT EXISTS runners (
  race_id TEXT NOT NULL REFERENCES races(race_id),
  horse_no INTEGER NOT NULL CHECK (horse_no > 0),
  horse_name TEXT NOT NULL,
  frame_no INTEGER,
  sex TEXT,
  age INTEGER CHECK (age IS NULL OR age >= 0),
  jockey TEXT,
  carried_weight REAL CHECK (carried_weight IS NULL OR carried_weight > 0),
  sire TEXT,
  dam TEXT,
  source TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  PRIMARY KEY (race_id, horse_no)
);

-- Store every observed quote as a time-series row; never overwrite an older quote.
CREATE TABLE IF NOT EXISTS odds_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  race_id TEXT NOT NULL REFERENCES races(race_id),
  bet_type TEXT NOT NULL,
  combination_key TEXT NOT NULL,
  odds REAL CHECK (odds IS NULL OR odds > 0),
  popularity_rank INTEGER CHECK (popularity_rank IS NULL OR popularity_rank > 0),
  captured_at TEXT NOT NULL,
  source TEXT NOT NULL,
  batch_id TEXT REFERENCES ingestion_batches(batch_id),
  UNIQUE (race_id, bet_type, combination_key, captured_at)
);

-- Immutable pre-race model outputs. Features are stored as JSON for schema evolution.
CREATE TABLE IF NOT EXISTS model_predictions (
  prediction_id TEXT PRIMARY KEY,
  race_id TEXT NOT NULL REFERENCES races(race_id),
  horse_no INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  feature_schema_version TEXT NOT NULL,
  rank_position INTEGER NOT NULL CHECK (rank_position > 0),
  score REAL NOT NULL,
  features_json TEXT NOT NULL,
  predicted_at TEXT NOT NULL,
  scheduled_post_at TEXT,
  minutes_to_post REAL,
  source TEXT NOT NULL,
  batch_id TEXT REFERENCES ingestion_batches(batch_id),
  FOREIGN KEY (race_id, horse_no) REFERENCES runners(race_id, horse_no),
  UNIQUE (race_id, horse_no, model_version, predicted_at)
);

CREATE TABLE IF NOT EXISTS official_results (
  race_id TEXT NOT NULL REFERENCES races(race_id),
  horse_no INTEGER NOT NULL,
  finish_position INTEGER CHECK (finish_position IS NULL OR finish_position > 0),
  finish_time_seconds REAL,
  last3f_seconds REAL,
  dead_heat_group TEXT,
  confirmed_at TEXT NOT NULL,
  source TEXT NOT NULL,
  batch_id TEXT REFERENCES ingestion_batches(batch_id),
  PRIMARY KEY (race_id, horse_no),
  FOREIGN KEY (race_id, horse_no) REFERENCES runners(race_id, horse_no)
);

CREATE TABLE IF NOT EXISTS payouts (
  race_id TEXT NOT NULL REFERENCES races(race_id),
  bet_type TEXT NOT NULL,
  combination_key TEXT NOT NULL,
  payout_yen INTEGER NOT NULL CHECK (payout_yen >= 0),
  stake_unit_yen INTEGER NOT NULL DEFAULT 100 CHECK (stake_unit_yen > 0),
  confirmed_at TEXT NOT NULL,
  source TEXT NOT NULL,
  batch_id TEXT REFERENCES ingestion_batches(batch_id),
  PRIMARY KEY (race_id, bet_type, combination_key)
);

CREATE INDEX IF NOT EXISTS idx_races_date_venue ON races(race_date, venue);
CREATE INDEX IF NOT EXISTS idx_races_type_date ON races(race_type, race_date);
CREATE INDEX IF NOT EXISTS idx_odds_race_time ON odds_snapshots(race_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_odds_type_time ON odds_snapshots(bet_type, captured_at);
CREATE INDEX IF NOT EXISTS idx_predictions_race_model ON model_predictions(race_id, model_version, predicted_at);
CREATE INDEX IF NOT EXISTS idx_results_finish ON official_results(race_id, finish_position);

INSERT OR IGNORE INTO schema_meta(version, applied_at, description)
VALUES (1, CURRENT_TIMESTAMP, 'Initial KEIBALAB research database schema');

INSERT OR IGNORE INTO schema_meta(version, applied_at, description)
VALUES (2, CURRENT_TIMESTAMP, 'Classify flat and Banei races separately');

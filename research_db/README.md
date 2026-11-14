# KEIBALAB Research Database (v1)

This is a research-only database foundation. It does not change the production app, production Supabase project, model weights, or betting logic.

## Initial storage choice

Start with SQLite: it is a real relational database, runs locally without a server or subscription, supports transactions and constraints, and can later be migrated to PostgreSQL. Do not commit a live database file to Git; commit schema and ingestion code, and persist database snapshots as versioned artifacts or move to a managed PostgreSQL service only after cost and retention are understood.

## Data model

- `races`: one row per local race identity (date + venue + race number).
- `runners`: one row per horse per race.
- `odds_snapshots`: append-only time series; old odds are never overwritten.
- `model_predictions`: immutable model/rank/score/features with model and feature versions.
- `official_results`: confirmed finishing positions and race-time facts.
- `payouts`: official payout by bet type and normalized combination.
- `ingestion_batches`: source, status, counts, checksum, and audit trail.
- `schema_meta`: schema version.

## Data integrity rules

1. Race identity is `race_date + venue + race_number`; the canonical key currently follows the existing `venue|YYYYMMDD|race_number` convention.
2. Horse number is scoped to a race; horse names are not identifiers.
3. Odds are time-series observations, not a mutable current-value column.
4. Pre-race predictions are immutable and keep the exact capture timestamp.
5. Odds, payouts, and current-race outcomes must never be features for the pre-race A45 prediction.
6. Result and payout rows are added only from official/verified sources; unknown values stay NULL, never guessed.
7. Every import is recorded in `ingestion_batches`; repeated imports must be idempotent.
8. Research database work stays isolated from `main` and production Supabase until separately reviewed.

## Current scope and next implementation steps

This commit establishes the schema only. Before creating a persistent database file or import job:
- validate schema with SQLite and foreign-key checks;
- map existing NAR monthly archive and daily racecard fields;
- write an idempotent importer with rejected-row reporting;
- import a small sample and reconcile counts against source files;
- add backup/restore and artifact-retention checks.

No automated polling schedule is enabled yet. No paid service is created.

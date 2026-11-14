# KEIBALAB Research Database (v1)

This is a research-only database foundation. It does not change the production app, production Supabase project, model weights, or betting logic.

## Initial storage choice

Start with SQLite: it is a real relational database, runs locally without a server or subscription, supports transactions and constraints, and can later be migrated to PostgreSQL. Do not commit a live database file to Git; commit schema and ingestion code, and persist database snapshots as versioned artifacts or move to a managed PostgreSQL service only after cost and retention are understood.

## Data model

- `races`: one row per local race identity (date + venue + race number), with explicit `race_type` = `flat` or `banei`.
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

## Historical official archive importer

The current schema adds a race-type discriminator so Banei records cannot be accidentally mixed into the flat-racing model dataset.

`scripts/import_nar_archive_to_research_db.py` reuses the existing NAR monthly ZIP parser and imports:
- race metadata and runners;
- official finishing position, time, and last-3F values;
- official payouts, including the 3-horse `trio` combination.

Example for one completed month:

```bash
python scripts/import_nar_archive_to_research_db.py \
  --start-date 2026-10-01 --end-date 2026-10-31 --max-months 1
```

The importer stores source checksums and batch IDs, upserts corrected race/result/payout rows, and is repeatable. `confirmed_at` in the historical result/payout tables currently records the source race date (day precision only); the monthly archive does not provide the exact time at which a result was confirmed.

Important separation:
- Banei (帯広ば) is stored in the same database but explicitly tagged `race_type='banei'`; flat-racing training/evaluation must filter to `race_type='flat'` unless a separate Banei model is being evaluated.
- Monthly archives contain final outcomes and payouts, not historical pre-race odds snapshots.
- Do not use final odds or payouts as model features.
- Historical odds snapshots and frozen pre-race predictions must be imported from their own timestamped sources.
- The importer does not write to production Supabase or alter the production app.

## Remaining work

- Run a real one-month archive import and reconcile race/runner/result/payout counts against the source.
- Expand to the full completed archive window only after the one-month reconciliation passes.
- Add backup/restore and durable local storage instructions; GitHub Actions artifacts are temporary.
- Design a separate daily capture import for timestamped odds and frozen predictions.

No automated polling schedule is enabled yet. No paid service is created.

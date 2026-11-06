alter table public.model_prediction_snapshots
  add column if not exists locked_payload jsonb;

update public.model_prediction_snapshots
set locked_payload = jsonb_build_object(
  'race_key', race_id,
  'date', to_char(race_date, 'YYYYMMDD'),
  'captured_at', captured_at,
  'scheduled_post_time', scheduled_post_time,
  'model_version', model_version,
  'model_temperature', model_temperature,
  'predictions', predictions,
  'candidate_combinations', candidate_combinations,
  'prediction_lock', prediction_lock
)
where locked_payload is null;

alter table public.model_prediction_snapshots
  alter column locked_payload set not null;

alter table public.model_prediction_snapshots
  drop constraint if exists model_prediction_snapshots_locked_payload_check;
alter table public.model_prediction_snapshots
  add constraint model_prediction_snapshots_locked_payload_check
  check (jsonb_typeof(locked_payload) = 'object');

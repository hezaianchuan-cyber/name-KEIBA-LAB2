create or replace function public.keiba_storage_health()
returns jsonb
language plpgsql
security definer
set search_path = public, storage
as $$
declare
  v jsonb;
begin
  select jsonb_build_object(
    'database_bytes', pg_database_size(current_database())::bigint,
    'database_mb', round(pg_database_size(current_database()) / 1048576.0, 2),
    'odds_table_bytes', pg_total_relation_size('public.odds_snapshots')::bigint,
    'odds_rows', (select count(*) from public.odds_snapshots)::bigint,
    'archive_bucket_bytes', coalesce((
      select sum(coalesce(nullif(metadata->>'size','')::bigint, 0))
      from storage.objects where bucket_id = 'keiba-odds-archive'
    ), 0)::bigint,
    'quota_reference_bytes', 524288000,
    'quota_reference', '500 MiB reference only; verify actual project plan quota',
    'warning_at_bytes', 367001600,
    'critical_at_bytes', 445644800,
    'status', case
      when pg_database_size(current_database()) >= 445644800 then 'critical'
      when pg_database_size(current_database()) >= 367001600 then 'warning'
      else 'normal'
    end
  ) into v;
  insert into public.keiba_storage_health_log (
    database_bytes, odds_table_bytes, odds_rows, archive_bucket_bytes,
    quota_reference_bytes, status
  ) values (
    (v->>'database_bytes')::bigint, (v->>'odds_table_bytes')::bigint,
    (v->>'odds_rows')::bigint, (v->>'archive_bucket_bytes')::bigint,
    (v->>'quota_reference_bytes')::bigint, v->>'status'
  );
  return v;
end;
$$;
revoke all on function public.keiba_storage_health() from public, anon, authenticated;
grant execute on function public.keiba_storage_health() to service_role;

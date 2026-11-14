# Local persistence and daily sync

## What this provides

- A real SQLite file stored on your own PC, not an expiring GitHub artifact.
- A daily Windows Task Scheduler job that fetches official pre-off odds snapshots from origin/main.
- Frozen live-model and Model-G predictions plus odds stored with capture timestamps. A45 research predictions are imported separately when available.
- Repeated runs do not duplicate prediction or odds rows.
- No Supabase writes, no paid database, and no writes back to GitHub.

The existing nar-preoff-odds.yml workflow already captures and commits timestamped daily odds snapshots to main every five minutes. This local task consumes those archives; it does not create a second collector.

## One-time setup on Windows

1. Install Git for Windows and Python 3.11+.
2. Clone the repository and switch to the research branch:

   git clone --branch research/keibalab-db-foundation https://github.com/hezaianchuan-cyber/name-KEIBA-LAB2.git
   cd name-KEIBA-LAB2

   If the repository is private, sign in with your normal GitHub credential manager. Never paste a personal access token into scripts or chat.

3. Download the keibalab-research-sqlite-four-years artifact from this successful build:
https://github.com/hezaianchuan-cyber/name-KEIBA-LAB2/actions/runs/43581365448

   Extract keibalab-research.sqlite into the clone's data folder. The exact path must be data/keibalab-research.sqlite.

4. Run the initial sync:
   python scripts/sync_research_db_local.py

5. Register the daily local task:
   powershell -ExecutionPolicy Bypass -File scripts/setup_research_db_task.ps1

   It runs at 02:15 local time. Windows must be on and the same account must be logged in. If the PC is off, Task Scheduler starts it at the next available opportunity after login.

## Backtest export

After syncing, run:
   python scripts/export_research_backtest_dataset.py --race-type flat

The exporter keeps predictions, odds, official results, and trio payouts in separate fields. For A45 it selects a trio-odds snapshot strictly after the frozen prediction and before post time. For live models whose predictions are generated from the same pre-off snapshot, it selects odds at or before the prediction timestamp. Incomplete races are marked partial and must not silently be counted as completed evaluation cases.

Output files:
- data/research-backtest-dataset.jsonl
- data/research-backtest-dataset.report.json

## Important limits

- GitHub Actions artifacts expire. They are only the transfer method for the initial 4-year database; after extraction, your local SQLite file is the durable copy.
- The daily local sync requires the PC to be on, GitHub access, and the Windows account to be logged in. It also imports any updated daily official race-result and payout files committed to main.
- The source collector is already scheduled on GitHub Actions, but collection can fail if the official NAR daily archive is unavailable. The sync only imports what was actually committed.
- The current daily collector stores odds snapshots and frozen live predictions. A45 research predictions are imported when the A45 capture workflow has produced them; that separate workflow is not automatically scheduled by this setup.
- This does not backfill historical intraday odds for 2022-2026. Monthly official archives provide results and payouts; timestamped odds coverage starts only where daily snapshot archives exist.
- Do not use final results or payouts as pre-race model features. The exporter is for retrospective evaluation only.
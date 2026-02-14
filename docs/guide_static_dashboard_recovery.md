# Static Dashboard Recovery Guide

**Tag**: `v0.1-static-dashboard`
**Commit**: `939e7ca`
**Date Tagged**: 2026-02-14

## What This Tag Contains

The fully working static dashboard with 9 analyses, before the SQLite refactoring:

| # | Analysis | Status |
|---|----------|--------|
| 1 | Investment Clock Sectors | 4-regime composite |
| 2 | SPY vs RETAILIRSA | Single indicator |
| 3 | SPY vs INDPRO | Single indicator |
| 4 | XLRE vs Orders/Inventories | Single indicator |
| 5 | XLP vs RETAILIRSA | Single indicator |
| 6 | XLY vs RETAILIRSA | Single indicator |
| 7 | XLRE vs New Home Sales | Single indicator, lag +8 |
| 8 | XLI vs ISM Manufacturing PMI | Single indicator, confirmatory |
| 9 | XLI vs ISM Services PMI | Single indicator, confirmatory |

7 pages per analysis: Overview, Qualitative, Correlation, Lead-Lag, Regimes, Backtests, Forecasts.

## How to View the Static Dashboard (Read-Only)

```bash
# View the code at that point in time (detached HEAD)
git checkout v0.1-static-dashboard

# Browse files, run dashboard, etc.
# When done, return to current branch:
git checkout main
```

## How to Restore as a Working Branch

```bash
# Create a new branch from the tag
git checkout -b static-dashboard-restore v0.1-static-dashboard

# Run the dashboard from this branch
docker compose build && docker compose up -d
```

## How to Run the Static Dashboard Without Switching Branches

```bash
# Export the tagged version to a separate folder
mkdir -p /tmp/rlic-static
git archive v0.1-static-dashboard | tar -x -C /tmp/rlic-static

# Run from that folder
cd /tmp/rlic-static
docker compose build && docker compose up -d
```

## How to Compare Static vs Refactored

```bash
# See what changed between the static version and current
git diff v0.1-static-dashboard..main --stat

# See changes in a specific file
git diff v0.1-static-dashboard..main -- src/dashboard/navigation.py
```

## Key Files in the Static Version

```
src/dashboard/
├── Home.py                          # Hardcoded 9-card grid
├── navigation.py                    # ANALYSES dict (single source of truth)
├── data_loader.py                   # existing_files dict + elif fallbacks
├── analysis_engine.py               # 28 analysis-agnostic functions (unchanged in refactoring)
├── components.py                    # 12 reusable chart components (unchanged in refactoring)
└── pages/
    ├── 2_📊_Overview.py             # elif column detection (10 branches)
    ├── 3_📖_Qualitative.py          # elif content blocks (851 lines)
    ├── 4_📈_Correlation.py          # elif column detection (10 branches)
    ├── 5_🔄_Lead_Lag.py             # elif column detection (10 branches)
    ├── 6_🎯_Regimes.py              # elif column detection (10 branches)
    ├── 7_💰_Backtests.py            # elif column detection (10 branches)
    └── 8_🔮_Forecasts.py            # elif column detection (10 branches)
```

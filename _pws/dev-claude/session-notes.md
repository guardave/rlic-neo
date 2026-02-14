# Session Notes - Dev Claude

## Session 2026-02-14

### Work Completed

1. **Dashboard Refactoring: SQLite Config Database**
   - Created `src/dashboard/config_db.py` — SQLite-backed config replacing 64 elif branches
   - Created `script/seed_config_db.py` — idempotent seed script for analyses + indicators
   - Single `resolve_columns()` function handles all column detection for all 7 pages
   - Two contexts: 'default' (display pages) and 'trading' (Regimes/Backtests)

2. **Dynamic Lag Slider**
   - Regimes and Backtests pages now read lag parameters from config DB
   - Each analysis has configurable `default_lag`, `lag_min`, `lag_max`

3. **Qualitative Content Extraction**
   - Extracted inline qualitative content to `docs/qualitative/{analysis_id}.md`
   - Created admonition renderer (`render_qualitative_section()`) in components.py

4. **Docker Build + Integration**
   - Updated Dockerfile: copies qualitative content + seed script, runs seed during build
   - Updated docker-compose.dev.yml: mounts qualitative dir for hot-reload
   - Full integration test: health check 200, all pages 200, no errors

5. **Streamlit Deprecation Fixes**
   - Replaced `use_container_width=True` with `width='stretch'` (24 occurrences, 9 files)
   - Replaced `applymap()` with `map()` in Lead_Lag page

6. **Streamlit Cloud Deployment Fix**
   - Added `_ensure_seeded()` auto-seed mechanism in config_db.py
   - Initially only triggered when DB empty (count == 0)
   - Fixed to always re-seed to handle additions/removals in seed data

7. **Sidebar Dropdown Readability**
   - Removed `:material/icon:` prefix from dropdown items
   - Final: uses `analyses[x]['name']` (plain text, e.g., "XLP vs Retail Inv/Sales")

8. **Investment Clock Removal**
   - Removed from ANALYSES_SEED and INDICATORS_SEED
   - Deleted `docs/qualitative/investment_clock.md`
   - Fixed stale DB issue: `seed_analyses()` now DELETEs all before re-inserting

### Commits (10 in this session)
- `e5cdc62` Design doc for refactoring
- `939e7ca` Implementation plan (20 tasks, 6 phases)
- `8620cca` Static dashboard recovery guide (tag v0.1)
- `d523613` config_db.py + seed script
- `df23d36` Replace 64 elif chains with resolve_columns()
- `5c9c3a2` Dynamic lag slider
- `a2d4bd8` Qualitative content extraction
- `16468bd` Docker build + deprecation fixes
- `384a72f` Auto-seed for Streamlit Cloud
- `d25404a`..`47962b5` Dropdown readability + Investment Clock removal + stale DB fix

### Key Insights

1. **SQLite as Config Store** — Simple, zero-dependency config that works on Streamlit Cloud (no Redis/Postgres needed)
2. **Always re-seed on startup** — Checking `count == 0` misses schema/data changes; idempotent seed is cheap
3. **DELETE + INSERT > INSERT OR REPLACE** — The latter doesn't remove stale rows
4. **Streamlit Cloud has no build step** — Auto-seed at import time is the only reliable mechanism

### Context from Previous Sessions
- RA Cheryl: Added 9 analyses following SOP v1.3
- QA Keung: Reviewed SOP, all fixes verified in v1.2
- Dev Claude (2026-01-03): Fixed heatmap title bug

## Session 2026-01-03

### Work Completed

1. **Fixed Heatmap Title Inconsistency**
   - Issue: `plot_sector_heatmap()` function had hardcoded "1-month lag" in title
   - Both Lag=0 and Lag=1 heatmaps showed incorrect "1-month lag" label
   - Fix: Added `lag` parameter to function with dynamic title generation
   - Now correctly shows "Lag=0 (Control)" and "Lag=1 (Optimal)"

2. **Files Modified**
   - `script/sector_regime_analysis.py`: Updated `plot_sector_heatmap()` function signature and calls

### Key Insights

- Side-by-side comparison methodology is important for research validity
- Control (lag=0) vs Optimal (lag=1) provides clear baseline for evaluation
- Dynamic labeling prevents confusion in visualizations

### Context from Previous Session

The previous session established:
- Investment Clock Sector Analysis Framework (doc 12)
- Lag sensitivity analysis comparing Lag=0, 1, 2, 3
- Side-by-side presentation requirement for all sections
- Qualitative analysis of dimensions
- Lead-lag analysis framework

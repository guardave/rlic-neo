# Memories - Dev Claude

## Lessons Learned

### 2026-02-14

1. **SQLite Auto-Seed Strategy**
   - Always re-seed on startup for Streamlit Cloud; `count == 0` check misses data changes
   - Use DELETE + INSERT (not INSERT OR REPLACE) to handle removals
   - `importlib.util` for dynamic module loading in auto-seed

2. **Streamlit Cloud Deployment**
   - No Docker build step — DB must be created at runtime
   - `.db` files should be gitignored; auto-seed ensures consistency
   - App auto-deploys on GitHub push; takes ~1-2 minutes

3. **Streamlit Deprecation (2025-2026)**
   - `use_container_width=True` → `width='stretch'`
   - `applymap()` → `map()` (pandas 2.1+ Styler)

4. **Edit Tool with Emoji Filenames**
   - Can fail with "File has not been read yet" when editing in parallel
   - Workaround: re-read each file via Glob → Read before Edit

5. **Config-Driven Column Detection**
   - Pattern-based: `indicator_pattern` with pipe-separated alternatives
   - Exact-column: `indicator_columns` with JSON list
   - Filter-based: `indicator_filter` with contains/and_contains rules
   - Two contexts: 'default' and 'trading' (for lagged columns)

### 2026-01-03

1. **Dynamic Labels in Visualizations**
   - Always parameterize visualization titles when generating multiple variants
   - Hardcoded strings lead to confusing/incorrect outputs
   - Use descriptive labels like "Control" and "Optimal" alongside technical values

2. **Research Methodology Best Practices**
   - Always include control set (lag=0) alongside optimal configuration
   - Side-by-side comparison is more convincing than sequential presentation
   - Document methodology clearly in framework documents

### Project-Specific Knowledge

- **Investment Clock**: 4-phase regime model (Recovery, Overheat, Stagflation, Reflation) — removed from dashboard
- **Best Indicators**: Orders/Inventories Ratio (growth) + PPI (inflation)
- **Optimal Lag**: Lag=1 is recommended for realistic implementation
- **Classification Rate**: 96.8% with Orders/Inv + PPI vs 66% benchmark
- **Dashboard**: 8 analyses, 7 pages each, SQLite-driven config, Streamlit Cloud + Docker
- **Analyses in dashboard**: spy_retailirsa, spy_indpro, xlre_orders_inv, xlp_retailirsa, xly_retailirsa, xlre_newhomesales, xli_ism_mfg, xli_ism_svc

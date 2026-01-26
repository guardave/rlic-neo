# Memories - QA Keung

## Key Learnings from RLIC Project

### SOP Review Best Practices
1. **Categorize by severity** - High/Medium/Low helps prioritize response
2. **Code review matters** - Bare `except: pass` in documentation samples is still a code smell
3. **Statistical rigor gaps** - Multiple testing correction often missed in analysis SOPs
4. **Effect size vs significance** - p-value alone is insufficient; economic significance matters

### RLIC Project Specifics
- **Dashboard**: Streamlit-based, 7 analysis pages (not Dash as originally documented)
- **Data format**: Parquet files in `data/` folder
- **Analysis pairs**: SPY/XLRE/XLP/XLY vs various indicators (RETAILIRSA, INDPRO, etc.)
- **SOP location**: `docs/sop/unified_analysis_sop.md` (v1.2 as of 2026-01-26)

### Team Collaboration
- **RA Cheryl**: Research Analyst, primary SOP author, responsive to feedback
- **Dev Claude**: Developer, handles bug fixes and implementation
- **Status board**: `_pws/_team/status-board.md` for async communication

### Quality Checklist (from SOP v1.2)
- Phase 2: |r| >= 0.15 AND p < 0.10 to proceed
- Fast-fail: |r| < 0.10 AND p > 0.30 → skip to Phase 7
- WFER > 0.5 for strategy viability
- All 7 dashboard pages must render before delivery
- Docker verification required: `docker compose -f docker-compose.dev.yml up -d`

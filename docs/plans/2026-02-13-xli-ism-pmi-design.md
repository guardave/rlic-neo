# Design: XLI vs ISM Manufacturing PMI / ISM Services PMI

**Date:** 2026-02-13
**Author:** RA Cheryl
**Status:** Approved

## Overview

Two independent analyses exploring whether ISM PMI indicators have predictive power for the Industrials sector (XLI ETF). Each follows the full SOP v1.3 seven-phase pipeline.

| Analysis | Dashboard ID | Target | Indicator | FRED Series |
|----------|-------------|--------|-----------|-------------|
| XLI vs ISM Manufacturing PMI | `xli_ism_mfg` | XLI | ISM Manufacturing PMI | `NAPM` (fallback: `MANEMP`, `MMNRNJ`) |
| XLI vs ISM Services PMI | `xli_ism_svc` | XLI | ISM Services PMI | `NMFBAI` (fallback: `ISMNSA`) |

**Execution order:** Manufacturing first (stronger theoretical link to Industrials), then Services. Lessons from Manufacturing inform Services analysis.

## Economic Rationale

**ISM Manufacturing PMI:** Monthly survey of 300+ purchasing managers. Published 1st business day of month (timing advantage). PMI > 50 = expansion, <= 50 = contraction. Directly measures manufacturing activity — XLI holdings (Boeing, Caterpillar, Honeywell, GE Aerospace) are major manufacturers.

**ISM Services PMI:** Monthly survey of non-manufacturing firms. Published 3rd business day. Less direct link to XLI, but services sector health affects industrial supply chains and demand. Worth testing as an independent signal.

**Expected relationships:**
- Manufacturing PMI likely leads XLI by 1-3 months (survey captures real-time sentiment before production data)
- Services PMI may show weaker or no relationship with XLI (indirect linkage)

## Scope Per Analysis

Each analysis produces:

| Artifact | Path Pattern |
|----------|-------------|
| Script | `script/analyze_xli_ism_{mfg,svc}.py` |
| Main data | `data/xli_ism_{mfg,svc}_full.parquet` |
| Lead-lag data | `data/xli_ism_{mfg,svc}_leadlag.parquet` |
| Regime data | `data/xli_ism_{mfg,svc}_regimes.parquet` |
| Report | `docs/analysis_reports/xli_ism_{mfg,svc}_analysis.md` |

## SOP v1.3 Pipeline Per Analysis

| Phase | Actions | Key Parameters |
|-------|---------|----------------|
| 0 - Qualitative | Document indicator definition, source, release timing, economic rationale | Self-contained per analysis |
| 1 - Data Prep | Fetch FRED series + XLI. Compute Level, MoM, YoY, Direction, Z-Score. Regime labels. | min n >= 60, <20% missing |
| 2 - Statistical | Concurrent correlation (lag=0) | |r| >= 0.15, p < 0.05 for significance |
| 3 - Lead-Lag | Cross-correlation -18 to +18 months. **Fast-fail decision here.** | Fast-fail if best |r| < 0.10 AND p > 0.30 across ALL lags |
| 4 - Regime | PMI > 50 = Expansion, <= 50 = Contraction. XLI returns per regime. | t-test for regime difference |
| 5 - Backtest | Walk-forward (60m train, 12m test, 3m purge). Signal lag = 1 month. | WFER > 0.5, Monte Carlo seed=42 |
| 6 - Dashboard | Add to all 7 pages with column detection handlers | Docker test before delivery |
| 7 - Documentation | Full analysis report (positive or negative result) | |

## Data Specifications

**Column naming convention (with prefix to avoid collisions):**
- `ISM_Mfg_PMI_Level`, `ISM_Mfg_PMI_MoM`, `ISM_Mfg_PMI_YoY`, `ISM_Mfg_PMI_Direction`
- `ISM_Svc_PMI_Level`, `ISM_Svc_PMI_MoM`, `ISM_Svc_PMI_YoY`, `ISM_Svc_PMI_Direction`
- `XLI_Price`, `XLI_Returns`
- `Regime` (labeled "Manufacturing Expansion"/"Manufacturing Contraction" or "Services Expansion"/"Services Contraction")

**Data period:** XLI inception ~1998 to present. ISM Manufacturing available from 1948, Services from 1997. Effective range ~1998-present (~300 months).

**FRED series verification:** Script will try primary series first, fall back to alternatives if unavailable, and log which series was used.

## Dashboard Integration

**Navigation entries:**

```python
'xli_ism_mfg': {
    'name': 'XLI vs ISM Manufacturing PMI',
    'icon': '🏭',
    'short': 'XLI-MFG',
    'description': 'Industrials sector vs ISM Manufacturing PMI'
}
'xli_ism_svc': {
    'name': 'XLI vs ISM Services PMI',
    'icon': '🏢',
    'short': 'XLI-SVC',
    'description': 'Industrials sector vs ISM Services PMI'
}
```

**Dashboard count:** 7 -> 8 (after Manufacturing) -> 9 (after Services)

**Home page:** Add cards using current pattern. Dynamic card generation noted as future refactoring.

**Lead-Lag page:** Default slider range 12 months (ISM is faster-moving than housing).

**Regime labels:** Clearly labeled as "Manufacturing Expansion/Contraction" or "Services Expansion/Contraction" to distinguish between the two when switching.

## Known Issues and Mitigations

| Issue | Mitigation |
|-------|-----------|
| Home page card layout at 9 analyses | Current grid still works at 9. Dynamic catalog is a future refactor task. |
| Column detection elif chains growing | Use distinct prefixes. Config-driven refactor planned as separate task. |
| FRED series ID uncertainty | Fallback chain with logging. Verify at runtime. |
| XLI limited history (~1998) | ~300 months is adequate (min 60 per SOP). |
| Both analyses share XLI target | Each analysis is self-contained. No shared state. |

## Future Work (Out of Scope)

1. **Dashboard refactoring:** Replace elif handler chains with config-driven column mapping (Option A). Then evolve to convention-based auto-detection (Option C).
2. **Dynamic Home page:** Auto-generate cards from ANALYSES dict with responsive grid and category filtering.
3. **Additional PMI sub-indices:** New Orders, Employment, Production sub-components if headline PMI proves significant.

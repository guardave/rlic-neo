# RA Cheryl - Key Memories and Insights

## Critical Lessons

### 1. Fast-Fail Timing (SOP v1.3) - 2026-01-27
**The most important lesson from this session:**
- Fast-fail at concurrent correlation (lag=0) can miss significant relationships at other lags
- XLRE vs New Home Sales: r=0.06 at lag=0, but r=0.22 (p=0.025) at lag=+8
- **Always run full lead-lag analysis (-18 to +18) BEFORE fast-fail decision**

### 2. Extended Lead-Lag Range - 2026-01-27
- Default range of -12 to +12 may miss significant relationships
- Housing indicators often show 6-12+ month leads due to economic transmission delays
- Use -18 to +18 for comprehensive analysis, or -24 to +24 for housing-related indicators

### 3. Positive vs Negative Lags - 2026-01-27
- **Positive lag**: Indicator leads target (predictive, actionable for trading)
- **Negative lag**: Target leads indicator (reverse causality, not actionable)
- Building Permits showed only reverse causality - market anticipates housing data

### 4. Dashboard Page Updates - 2026-01-26
When adding a new analysis, ALL dashboard pages must be updated:
- navigation.py: Add entry to ANALYSES dict
- data_loader.py: Add file mapping
- Home.py: Add card and update count
- All 6 page files: Add column detection handlers

### 5. Effect Size vs Statistical Significance
- p < 0.05 alone is not enough
- |r| >= 0.15 is minimum for economic/practical significance
- Small correlations can be statistically significant with enough data but useless for trading

### 6. Negative Results Are Valid
- Document negative results properly
- They prevent false confidence in non-working strategies
- Example: Building Permits shows NO predictive power for XLRE at any lag

## Technical Notes

### Column Detection Pattern
```python
elif analysis_id == 'xlre_newhomesales':
    indicator_cols = ['NewHomeSales_YoY_Lagged'] if 'NewHomeSales_YoY_Lagged' in data.columns else ['NewHomeSales_YoY']
    return_cols = ['XLRE_Returns'] if 'XLRE_Returns' in data.columns else []
    if 'Regime' in data.columns and 'regime' not in data.columns:
        data['regime'] = data['Regime']
```

### Lead-Lag Default for Housing Analyses
```python
max_lag = st.slider("Max Lag (months)", 6, 24, 24 if analysis_id == 'xlre_newhomesales' else 12)
```

## Project Knowledge

### XLRE Housing Analysis Summary
| Indicator | Predictive Power | Best Lag | Action |
|-----------|-----------------|----------|--------|
| New Home Sales | YES | +8 months | Use as signal |
| Building Permits | NO | None | Do not use |

### Why +8 Month Lag for New Home Sales?
1. Home sales reflect buyer decisions from 3-6 months prior
2. Construction activity follows sales by 6-12 months
3. REIT revenue impacts show with multi-quarter delay
4. Market takes time to fully price in housing trends

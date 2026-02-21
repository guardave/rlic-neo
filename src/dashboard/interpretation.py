"""
Interpretation Engine for RLIC Dashboard.

Generates template-based textual annotations from analysis_results DB values.
Human overrides from analysis_annotations replace auto-generated text when present.
"""

from typing import Dict, Optional
from src.dashboard.config_db import (
    get_result, get_annotation
)


# =============================================================================
# Helpers
# =============================================================================

def _correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r < 0.10: return "negligible"
    if abs_r < 0.30: return "weak"
    if abs_r < 0.50: return "moderate"
    if abs_r < 0.70: return "strong"
    return "very strong"


def _correlation_direction(r: float) -> str:
    return "positive" if r > 0 else "negative"


def _p_qualifier(p: float) -> str:
    if p is None: return "N/A"
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return f"= {p:.3f}"
    if p < 0.05:  return f"= {p:.3f}"
    return f"= {p:.2f}"


def _format_r(r: float) -> str:
    return f"{r:.4f}" if r is not None else "N/A"


def _granger_direction_label(direction: str) -> str:
    labels = {
        'predictive': 'Predictive',
        'confirmatory': 'Confirmatory',
        'bi-directional': 'Bi-directional',
        'independent': 'Independent',
    }
    return labels.get(direction, direction)


# =============================================================================
# Section Interpreters — each returns {intro, finding, interpretation, verdict}
# =============================================================================

def _interp_overview_summary(analysis_id: str, indicator_name: str,
                              target_name: str) -> Dict[str, str]:
    r_level = get_result(analysis_id, 'correlation', 'pearson_r_level')
    r_change = get_result(analysis_id, 'correlation', 'pearson_r_change')
    granger_dir = get_result(analysis_id, 'granger', 'direction')
    regime_pval = get_result(analysis_id, 'regime', 't_test_pvalue')

    parts = []
    if r_level:
        rv = r_level['value']
        parts.append(f"{indicator_name} shows a **{_correlation_strength(rv)} "
                     f"{_correlation_direction(rv)}** level correlation (r = {_format_r(rv)}) with {target_name}.")
    if r_change:
        rv = r_change['value']
        parts.append(f"The MoM change correlation is **{_correlation_strength(rv)} "
                     f"{_correlation_direction(rv)}** (r = {_format_r(rv)}).")
    if granger_dir:
        label = _granger_direction_label(granger_dir.get('value_text', ''))
        parts.append(f"Granger causality classification: **{label}**.")
    if regime_pval:
        pv = regime_pval['value']
        sig = "highly significant" if pv < 0.01 else "significant" if pv < 0.05 else "not significant"
        parts.append(f"Regime performance difference is **{sig}** (p {_p_qualifier(pv)}).")

    return {
        'finding': ' '.join(parts) if parts else None,
        'intro': None,
        'interpretation': None,
        'verdict': None,
    }


def _interp_correlation_level(analysis_id: str, indicator_name: str,
                               target_name: str) -> Dict[str, str]:
    r = get_result(analysis_id, 'correlation', 'pearson_r_level')
    p = get_result(analysis_id, 'correlation', 'pearson_p_level')
    n = get_result(analysis_id, 'correlation', 'pearson_n_level')

    if not r:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    rv, pv, nv = r['value'], p['value'] if p else None, int(n['value']) if n else None

    finding = (f"The level correlation between {indicator_name} and {target_name} is "
               f"**r = {_format_r(rv)}** (p {_p_qualifier(pv)}, n = {nv}). "
               f"This is a **{_correlation_strength(rv)} {_correlation_direction(rv)}** relationship.")

    interpretation = ("Level correlations between trending series can be inflated by common trends. "
                      "The change-based analysis below controls for this.")

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_correlation_change(analysis_id: str, indicator_name: str,
                                target_name: str) -> Dict[str, str]:
    r = get_result(analysis_id, 'correlation', 'pearson_r_change')
    p = get_result(analysis_id, 'correlation', 'pearson_p_change')
    n = get_result(analysis_id, 'correlation', 'pearson_n_change')
    r_level = get_result(analysis_id, 'correlation', 'pearson_r_level')

    if not r:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    rv, pv, nv = r['value'], p['value'] if p else None, int(n['value']) if n else None

    finding = (f"The MoM change correlation is **r = {_format_r(rv)}** "
               f"(p {_p_qualifier(pv)}, n = {nv}), "
               f"which is **{_correlation_strength(rv)} {_correlation_direction(rv)}**.")

    interpretation = None
    if r_level:
        rl = r_level['value']
        if abs(rv) > abs(rl):
            interpretation = ("The change correlation is stronger than the level correlation, "
                              "confirming the relationship is not driven by spurious trend overlap.")
        elif abs(rv) < abs(rl) * 0.7:
            interpretation = ("The change correlation is weaker than the level correlation, "
                              "suggesting some of the level relationship may be driven by common trends.")

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_correlation_rolling(analysis_id: str, indicator_name: str,
                                 target_name: str) -> Dict[str, str]:
    mean = get_result(analysis_id, 'correlation', 'rolling_corr_mean')
    std = get_result(analysis_id, 'correlation', 'rolling_corr_std')
    mn = get_result(analysis_id, 'correlation', 'rolling_corr_min')
    mx = get_result(analysis_id, 'correlation', 'rolling_corr_max')

    if not mean:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    finding = (f"The rolling correlation averages **{_format_r(mean['value'])}** "
               f"with range [{_format_r(mn['value'])} to {_format_r(mx['value'])}].")

    if std and std['value'] > 0.3:
        interpretation = ("The high variability indicates the relationship is **not stable** over time. "
                          "Periods where the correlation flips sign suggest structural regime changes.")
    elif std and std['value'] > 0.15:
        interpretation = "The relationship shows **moderate variability** over time."
    else:
        interpretation = "The relationship is **relatively stable** over the sample period."

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_leadlag_crosscorr(analysis_id: str, indicator_name: str,
                                target_name: str) -> Dict[str, str]:
    opt_lag = get_result(analysis_id, 'leadlag', 'optimal_lag')
    opt_r = get_result(analysis_id, 'leadlag', 'optimal_lag_r')
    sig_lags = get_result(analysis_id, 'leadlag', 'significant_lags')

    if not opt_lag:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    lag_val = int(opt_lag['value'])
    r_val = opt_r['value'] if opt_r else None
    n_sig = len(sig_lags['metadata']) if sig_lags and sig_lags.get('metadata') else 0

    finding = (f"The strongest cross-correlation is at **lag {lag_val}** "
               f"(r = {_format_r(r_val)}). "
               f"{n_sig} lag(s) are statistically significant (p < 0.05).")

    if lag_val > 0:
        interpretation = (f"{indicator_name} **leads** {target_name} by {lag_val} month(s). "
                          "This suggests predictive potential.")
    elif lag_val < 0:
        interpretation = (f"{target_name} **leads** {indicator_name} by {abs(lag_val)} month(s). "
                          "The target moves before the indicator (confirmatory signal).")
    else:
        interpretation = ("The relationship is **contemporaneous** — movements occur simultaneously "
                          "with no leading signal detected.")

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_granger_fwd(analysis_id: str, indicator_name: str,
                         target_name: str) -> Dict[str, str]:
    best_p = get_result(analysis_id, 'granger', 'fwd_best_pvalue')
    best_lag = get_result(analysis_id, 'granger', 'fwd_best_lag')
    best_f = get_result(analysis_id, 'granger', 'fwd_best_fstat')

    if not best_p:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    intro = f"**Does {indicator_name} Granger-cause {target_name}?**"
    pv = best_p['value']
    if pv < 0.05:
        finding = (f"Yes. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} "
                   f"(F = {best_f['value']:.2f}).")
        interpretation = (f"Past values of {indicator_name} contain information that helps "
                          f"predict future {target_name} beyond what past {target_name} alone can explain.")
    else:
        finding = f"No. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} (not significant at 0.05)."
        interpretation = (f"Past {indicator_name} values do not add predictive power for {target_name} "
                          f"beyond what past {target_name} already provides.")

    return {'intro': intro, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_granger_rev(analysis_id: str, indicator_name: str,
                         target_name: str) -> Dict[str, str]:
    best_p = get_result(analysis_id, 'granger', 'rev_best_pvalue')
    best_lag = get_result(analysis_id, 'granger', 'rev_best_lag')
    best_f = get_result(analysis_id, 'granger', 'rev_best_fstat')

    if not best_p:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    intro = f"**Does {target_name} Granger-cause {indicator_name}?**"
    pv = best_p['value']
    if pv < 0.05:
        finding = (f"Yes. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} "
                   f"(F = {best_f['value']:.2f}).")
        interpretation = (f"There is a feedback effect where {target_name} movements help predict "
                          f"subsequent {indicator_name} changes.")
    elif pv < 0.10:
        finding = (f"Weak evidence. Best p-value = {pv:.4f} at lag {int(best_lag['value'])} — "
                   "suggestive but not significant at 0.05.")
        interpretation = (f"There may be a mild feedback where {target_name} influences subsequent "
                          f"{indicator_name} changes, but the evidence is not conclusive.")
    else:
        finding = f"No. Best p-value = {pv:.4f} (not significant)."
        interpretation = f"Past {target_name} values do not help predict {indicator_name}."

    return {'intro': intro, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


def _interp_granger_verdict(analysis_id: str, indicator_name: str,
                             target_name: str) -> Dict[str, str]:
    direction = get_result(analysis_id, 'granger', 'direction')
    if not direction:
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    d = direction.get('value_text', 'unknown')
    label = _granger_direction_label(d)

    verdicts = {
        'predictive': (f"{indicator_name} is a **leading signal** for {target_name}. "
                       "Past indicator values help predict future target returns."),
        'confirmatory': (f"{target_name} moves first — {indicator_name} **confirms** afterward. "
                         "The indicator is useful for regime classification but not for timing entries."),
        'bi-directional': (f"The relationship is **bi-directional** — both {indicator_name} and "
                           f"{target_name} influence each other. The indicator has predictive power "
                           "but is also influenced by the target (feedback loop)."),
        'independent': (f"**No Granger-causal relationship** detected at monthly frequency. "
                        "Any correlation is contemporaneous, not predictive."),
    }

    return {
        'intro': None,
        'finding': f"**Verdict: {label}**",
        'interpretation': verdicts.get(d, ''),
        'verdict': verdicts.get(d, ''),
    }


def _interp_regime_performance(analysis_id: str, indicator_name: str,
                                target_name: str) -> Dict[str, str]:
    perf = get_result(analysis_id, 'regime', 'perf_summary')
    t_pval = get_result(analysis_id, 'regime', 't_test_pvalue')

    if not perf or not perf.get('metadata'):
        return {'intro': None, 'finding': None, 'interpretation': None, 'verdict': None}

    regimes = perf['metadata']
    parts = []
    for regime_name, regime_stats in regimes.items():
        mean_r = regime_stats.get('mean', 0)
        sharpe = regime_stats.get('sharpe', 0)
        parts.append(f"**{regime_name}**: mean return {mean_r*100:.2f}%/mo (Sharpe {sharpe:.2f})")

    finding = "Regime performance: " + " vs ".join(parts) + "."

    interpretation = None
    if t_pval:
        pv = t_pval['value']
        if pv < 0.01:
            interpretation = f"The regime performance difference is **highly significant** (p {_p_qualifier(pv)})."
        elif pv < 0.05:
            interpretation = f"The regime performance difference is **significant** (p {_p_qualifier(pv)})."
        else:
            interpretation = f"The regime performance difference is **not statistically significant** (p {_p_qualifier(pv)})."

    return {'intro': None, 'finding': finding, 'interpretation': interpretation, 'verdict': None}


# =============================================================================
# Registry: section_key -> interpreter function
# =============================================================================

_INTERPRETERS = {
    'overview.summary': _interp_overview_summary,
    'correlation.level': _interp_correlation_level,
    'correlation.change': _interp_correlation_change,
    'correlation.rolling': _interp_correlation_rolling,
    'leadlag.crosscorr': _interp_leadlag_crosscorr,
    'leadlag.granger_fwd': _interp_granger_fwd,
    'leadlag.granger_rev': _interp_granger_rev,
    'leadlag.granger_verdict': _interp_granger_verdict,
    'regime.performance': _interp_regime_performance,
}


# =============================================================================
# Public API
# =============================================================================

def get_interpretation(analysis_id: str, section_key: str,
                       indicator_name: str = "Indicator",
                       target_name: str = "Target") -> Dict[str, Optional[str]]:
    """
    Get interpretation text for a dashboard section.

    Resolution order:
    1. Check analysis_annotations for human override
    2. For any NULL fields, auto-generate from analysis_results + templates
    3. Return dict with {intro, finding, interpretation, verdict}
    """
    # Auto-generate
    interpreter = _INTERPRETERS.get(section_key)
    auto = interpreter(analysis_id, indicator_name, target_name) if interpreter else {}

    # Check for human override
    override = get_annotation(analysis_id, section_key)
    if override:
        for field in ('intro', 'finding', 'interpretation', 'verdict'):
            if override.get(field):
                auto[field] = override[field]

    return {
        'intro': auto.get('intro'),
        'finding': auto.get('finding'),
        'interpretation': auto.get('interpretation'),
        'verdict': auto.get('verdict'),
    }


def render_annotation(analysis_id: str, section_key: str,
                      indicator_name: str = "Indicator",
                      target_name: str = "Target") -> None:
    """
    Render interpretation text in Streamlit.

    Convenience function that calls get_interpretation() and renders
    non-None fields as markdown.
    """
    import streamlit as st

    interp = get_interpretation(analysis_id, section_key, indicator_name, target_name)

    if interp.get('intro'):
        st.markdown(interp['intro'])
    if interp.get('finding'):
        st.markdown(interp['finding'])
    if interp.get('interpretation'):
        st.markdown(f"*{interp['interpretation']}*")
    if interp.get('verdict'):
        st.success(interp['verdict'])

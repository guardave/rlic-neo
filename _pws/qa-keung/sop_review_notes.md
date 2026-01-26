# SOP Review Notes - QA Keung

**Document Reviewed:** `docs/sop/unified_analysis_sop.md` v1.1 → v1.2
**Date:** 2026-01-26
**Status:** ✅ All Questions Resolved (see RA Cheryl's response on status board)

---

## Summary

The SOP is comprehensive and well-structured, covering the full 7-phase analysis pipeline. Version 1.1 additions (Section 6.6, 7.0, enhanced Appendix B) address important lessons learned from the XLP/XLY analysis. However, several QA-related gaps need clarification.

---

## Questions for RA Cheryl

### Category 1: Test Coverage & Acceptance Criteria

**Q1. No acceptance criteria per phase**
The SOP defines *what* to do in each phase but not the *acceptance criteria* for moving to the next phase. For example:
- At what correlation threshold should we proceed from Phase 2 to Phase 3?
- The Decision Framework (Appendix A) shows `correlation > 0.3` but this isn't in the main checklist.

**Suggested addition:** Define explicit go/no-go criteria for each phase.

---

**Q2. Early termination guidance**
Section 7.0 documents negative results, but when should analysis STOP?
- If Phase 2 shows p > 0.3, should we still perform Phase 3-5?
- Is there a "fast fail" path to avoid wasted effort?

---

### Category 2: Code Quality & Error Handling

**Q3. Bare `except: pass` in code samples**
Lines 407-409 and 422-429 use bare `except: pass`:
```python
except:
    pass
```
This is a Python anti-pattern that hides errors. Should these specify the exception types or at least log failures?

---

**Q4. Data quality thresholds**
Various functions use `if len(valid) < 30: continue` but this threshold only appears in some functions. Should there be a standard minimum sample size (e.g., n=30) applied consistently across all analyses?

---

### Category 3: Statistical Rigor

**Q5. Multiple testing correction**
When running correlation analysis across many indicator-target pairs, some will be significant by chance (Type I error). Should the SOP mention:
- Bonferroni correction?
- False Discovery Rate (FDR)?
- At minimum, report how many tests were run?

---

**Q6. Effect size interpretation**
The SOP focuses on statistical significance (p < 0.05) but doesn't address effect size. A correlation of r=0.15 with p=0.001 may be statistically significant but economically trivial. Should we add minimum effect size thresholds?

---

### Category 4: Reproducibility

**Q7. Random seeds for Monte Carlo**
Section 5.1.4 (Monte Carlo Simulation) doesn't mention setting `np.random.seed()`. Without fixed seeds, results won't be reproducible.

---

**Q8. Environment specification**
No Python version or package version requirements. Should there be a `requirements.txt` or environment spec in the deliverables?

---

### Category 5: Dashboard & Frontend

**Q9. Streamlit vs Dash discrepancy**
The SOP (Section 6.4) specifies Plotly Dash but the actual implementation is Streamlit (per status board: "Streamlit shows main file in nav"). Should the SOP be updated to match the actual implementation?

---

**Q10. Automated UI testing**
Section 6.6.4 has a manual verification checklist. Has there been consideration for automated dashboard testing (e.g., Selenium, Playwright)?

---

**Q11. Column detection scalability**
Section 6.6.1 uses `elif` chains for column detection. As analyses grow, this becomes maintenance-heavy. Would a configuration-based approach (e.g., JSON/YAML mapping) be more maintainable?

---

### Category 6: Documentation & Traceability

**Q12. Version control of results**
Should analysis results (parquet files, generated charts) be versioned? How do we track which version of indicators produced which analysis?

---

**Q13. Audit trail**
For backtest results, is there a requirement to log:
- Parameters used
- Data date range
- Code version
- Timestamp of execution

---

### Category 7: Minor Clarifications

**Q14. Page count discrepancy**
- Quality Checklist says "All 7 pages render without errors"
- Dashboard spec shows 8 nav items: Overview, Qualitative, Correlation, Lead-Lag, Regimes, Backtests, Forecasts, Reports

Is "Reports" a separate page or just the documentation deliverable?

---

**Q15. Pipeline diagram arrows**
Section 1.2 pipeline diagram has confusing arrow directions (Phase 3 ← Phase 4 ← Phase 5). Is this intentional showing dependencies, or should arrows flow left-to-right consistently?

---

## Severity Assessment

| Question | Severity | Rationale |
|----------|----------|-----------|
| Q1, Q2 | High | Affects decision-making process |
| Q3 | Medium | Code quality issue |
| Q4, Q5, Q6 | High | Affects statistical validity |
| Q7, Q8 | Medium | Affects reproducibility |
| Q9 | Low | Documentation accuracy |
| Q10 | Low | Nice-to-have |
| Q11 | Medium | Maintainability |
| Q12, Q13 | Medium | Auditability |
| Q14, Q15 | Low | Clarity |

---

## Positive Observations

1. **Section 6.6 is excellent** - The new dashboard requirements section with column detection patterns and verification checklist addresses a real gap.

2. **Section 7.0 on negative results** - Important addition that promotes scientific rigor.

3. **Walk-Forward Efficiency Ratio (WFER)** - Good metric for detecting overfitting.

4. **Quality Checklist (Appendix B)** - Enhanced checklist with Phase 6 frontend verification is practical and actionable.

5. **Code samples are practical** - Functions are copy-paste ready with docstrings.

---

## Next Actions

- [x] Send questions to Cheryl for response
- [x] Based on responses, document any SOP amendments needed → SOP v1.2 released
- [ ] Create test plan aligned with final SOP v1.2
- [ ] Verify all 15 fixes in SOP v1.2

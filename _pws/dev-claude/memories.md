# Memories - Dev Claude

## Lessons Learned

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

- **Investment Clock**: 4-phase regime model (Recovery, Overheat, Stagflation, Reflation)
- **Best Indicators**: Orders/Inventories Ratio (growth) + PPI (inflation)
- **Optimal Lag**: Lag=1 is recommended for realistic implementation
- **Classification Rate**: 96.8% with Orders/Inv + PPI vs 66% benchmark

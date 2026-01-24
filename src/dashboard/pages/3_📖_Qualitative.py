"""
Qualitative Analysis Page.

Displays indicator profiles, economic interpretation, and literature references.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Qualitative Analysis", page_icon="📖", layout="wide")

# Get selected analysis from session state
analysis_id = st.session_state.get('selected_analysis', 'investment_clock')

st.title("📖 Qualitative Analysis")

# Analysis selector
analysis_options = {
    'investment_clock': 'Investment Clock Sector Analysis',
    'spy_retailirsa': 'SPY vs RETAILIRSA',
    'spy_indpro': 'SPY vs Industrial Production',
    'xlre_orders_inv': 'XLRE vs Orders/Inventories'
}

selected = st.selectbox(
    "Select Analysis",
    options=list(analysis_options.keys()),
    format_func=lambda x: analysis_options[x],
    index=list(analysis_options.keys()).index(analysis_id)
)

if selected != analysis_id:
    st.session_state['selected_analysis'] = selected
    st.rerun()

st.divider()

# ============================================================================
# Qualitative Content for Each Analysis
# ============================================================================

if analysis_id == 'investment_clock':
    st.header("Investment Clock Framework")

    st.subheader("Overview")
    st.markdown("""
    The Investment Clock framework evaluates **sector performance across four economic regimes** using:
    - **Growth Signal**: Orders/Inventories Ratio (3MA vs 6MA direction)
    - **Inflation Signal**: PPI (3MA vs 6MA direction)

    Together these achieve a 96.8% classification rate vs 66% with traditional indicators.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Growth Dimension")
        st.markdown("""
        **What Does "Growth" Mean?**

        In the Investment Clock context, "Growth" refers to the direction of economic activity—whether
        GDP, employment, and corporate earnings are accelerating or decelerating.
        """)

        st.markdown("**How Growth Affects Sectors:**")
        growth_table = pd.DataFrame({
            'Growth Direction': ['Rising', 'Falling'],
            'Sector Impact': ['Cyclicals outperform', 'Defensives outperform'],
            'Mechanism': [
                'Increased consumer spending, capital investment, hiring',
                'Stable demand for necessities; flight to safety'
            ]
        })
        st.dataframe(growth_table, hide_index=True, use_container_width=True)

        st.markdown("""
        **Growth-Sensitive Sectors** (High Beta to Growth):
        - Technology: Discretionary IT spending expands/contracts with growth
        - Consumer Discretionary: Durable goods, travel, entertainment
        - Industrials: Capital expenditure, manufacturing orders
        - Financials: Loan demand, credit quality

        **Growth-Defensive Sectors** (Low Beta to Growth):
        - Utilities: Regulated returns, inelastic demand
        - Consumer Staples: Food, beverages, household products
        - Healthcare: Non-discretionary spending
        """)

    with col2:
        st.subheader("Inflation Dimension")
        st.markdown("""
        **What Does "Inflation" Mean?**

        In the Investment Clock context, "Inflation" refers to the direction of price pressure—whether
        prices are accelerating or decelerating.
        """)

        st.markdown("**How Inflation Affects Sectors:**")
        inflation_table = pd.DataFrame({
            'Inflation Direction': ['Rising', 'Falling'],
            'Sector Impact': ['Real assets outperform', 'Rate-sensitive sectors outperform'],
            'Mechanism': [
                'Commodity producers benefit; pricing power matters',
                'Lower rates boost valuations; borrowing costs fall'
            ]
        })
        st.dataframe(inflation_table, hide_index=True, use_container_width=True)

        st.markdown("""
        **Inflation-Beneficiary Sectors** (Positive Beta to Inflation):
        - Energy: Direct commodity exposure; oil/gas price correlation
        - Materials: Mining, chemicals, commodity producers

        **Inflation-Hurt Sectors** (Negative Beta to Inflation):
        - Utilities: Regulated prices lag inflation; rising rates hurt
        - Consumer Discretionary: Purchasing power erosion
        """)

    st.subheader("Interaction Effects: Why Four Phases Matter")
    st.markdown("""
    The Investment Clock framework recognizes that growth and inflation **interact**:
    """)

    phase_effects = pd.DataFrame({
        'Growth': ['Rising', 'Rising', 'Falling', 'Falling'],
        'Inflation': ['Falling', 'Rising', 'Rising', 'Falling'],
        'Phase': ['Recovery', 'Overheat', 'Stagflation', 'Reflation'],
        'Combined Effect': [
            'Best for cyclicals - Growth boosts earnings; low inflation allows Fed accommodation',
            'Real assets - Growth supports demand; inflation boosts commodity prices',
            'Worst combo - No growth + price pressure = margin compression',
            'Rate-sensitive recovery - Fed eases; rate-sensitive sectors benefit'
        ]
    })
    st.dataframe(phase_effects, hide_index=True, use_container_width=True)

    st.subheader("Sector Sensitivity Matrix")
    sensitivity = pd.DataFrame({
        'Sector': ['Technology', 'Financials', 'Healthcare', 'Energy', 'Industrials',
                   'Consumer Disc.', 'Consumer Staples', 'Utilities', 'Materials'],
        'Growth Sensitivity': ['High (+)', 'High (+)', 'Low', 'Moderate (+)', 'High (+)',
                               'High (+)', 'Low (-)', 'Low (-)', 'Moderate (+)'],
        'Inflation Sensitivity': ['Moderate (-)', 'Mixed', 'Low', 'High (+)', 'Moderate (+)',
                                  'Moderate (-)', 'Low', 'High (-)', 'High (+)'],
        'Best Phase': ['Recovery', 'Recovery', 'Stagflation', 'Overheat', 'Overheat',
                       'Recovery/Reflation', 'Stagflation', 'Stagflation', 'Overheat'],
        'Worst Phase': ['Stagflation', 'Stagflation', '—', 'Reflation', 'Stagflation',
                        'Stagflation', 'Recovery', 'Overheat', 'Reflation']
    })
    st.dataframe(sensitivity, hide_index=True, use_container_width=True)

    st.subheader("Key Literature")
    st.markdown("""
    - **Fama (1981)** established the relationship between real economic activity and stock returns
    - **Chen, Roll & Ross (1986)** identified industrial production growth as a priced factor
    - **Boudoukh & Richardson (1993)** found inflation hedging varies by sector
    - **Invesco Inflation Research** documents sector rotation strategies
    """)

elif analysis_id == 'spy_retailirsa':
    st.header("Retail Inventories-to-Sales Ratio (RETAILIRSA)")

    st.subheader("What is the Retail Inventories-to-Sales Ratio?")
    st.markdown("""
    The Retail Inventories-to-Sales Ratio (RETAILIRSA) measures the relationship between end-of-month
    inventory values and monthly sales for retail businesses. Published monthly by the U.S. Census Bureau.

    **Key Characteristics:**
    - **Formula**: Inventory Level / Net Sales
    - **Interpretation**: A ratio of 1.5 means retailers hold 1.5 months of inventory relative to sales
    - **Historical Range**: Record high of 1.75 (April 1995), record low of 1.09 (June 2021)
    - **Current Median**: ~1.49
    - **Release Timing**: Mid-month, approximately 6 weeks after the reference month
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Signal Interpretation")
        signal_table = pd.DataFrame({
            'Ratio Level': ['Rising', 'Falling', 'High (>1.5)', 'Low (<1.3)'],
            'Interpretation': [
                'Retailers overstocked; demand weakening; potential slowdown',
                'Strong consumer demand; supply constraints; economic strength',
                'Excess inventory; potential markdowns; margin pressure',
                'Lean inventory; supply chain stress or strong demand'
            ]
        })
        st.dataframe(signal_table, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Role as a Recession Indicator")
        st.markdown("""
        Inventory levels as a recession signal:

        > *"In the past three recessions, Total Business Inventories reached 12-, 24-, and 36-month
        highs within 6 months of the recession's official start."* - NetSuite Research

        The self-reinforcing cycle: reduced consumer confidence → lower spending → rising inventories
        → production cuts → job losses → further confidence decline.
        """)

    st.subheader("The Bullwhip Effect Connection")
    st.markdown("""
    The ratio is closely tied to the **bullwhip effect** - a supply chain phenomenon where small changes
    in consumer demand create amplified swings upstream:

    > *"When businesses notice signs of an impending recession, such as rising inflation, interest rate
    hikes, and a slowdown in consumer spending, the bullwhip effect may be close behind."* - TrueCommerce

    The "inventory accelerator" explains why when demand decreases, the decline in orders is larger
    than the decline in sales as firms attempt to reduce their inventory levels.
    """)

    st.subheader("Key Insights from Literature")
    insights = pd.DataFrame({
        'Finding': [
            'Rising inventories precede recessions',
            'Ratio spikes within 6 months of recession',
            'Bullwhip effect amplifies small demand changes',
            'Record low in 2021 was anomaly',
            'Consumer spending drives inventory cycles'
        ],
        'Source': ['NetSuite, FocusEconomics', 'Historical analysis', 'MIT Sloan',
                   'FRED historical data', 'Morningstar'],
        'Implication': [
            'Early warning signal',
            'Useful for timing',
            'Explains volatility',
            'Context matters',
            'Demand is key driver'
        ]
    })
    st.dataframe(insights, hide_index=True, use_container_width=True)

    st.subheader("Limitations as a Stock Market Indicator")
    st.markdown("""
    1. **Publication Lag**: Data released ~6 weeks after month-end; markets have moved
    2. **Sector-Specific**: Retail represents portion of economy; doesn't capture services
    3. **Supply Chain Complexity**: Global supply chains complicate interpretation
    4. **Structural Changes**: E-commerce and just-in-time practices have lowered "normal" ratios
    5. **COVID Distortions**: 2020-2022 created unprecedented volatility in the series
    """)

elif analysis_id == 'spy_indpro':
    st.header("Industrial Production Index (INDPRO)")

    st.subheader("What is the Industrial Production Index?")
    st.markdown("""
    The Industrial Production Index (INDPRO) measures the real output of all relevant establishments
    located in the United States for manufacturing, mining, and electric and gas utilities. Published
    monthly by the Federal Reserve Board since 1919.

    **Key Characteristics:**
    - **Coincident Indicator**: Moves with the business cycle, not ahead of it
    - **Coverage**: ~20% of the U.S. economy (manufacturing-focused)
    - **Release Timing**: Mid-month, approximately 15 days after the reference month
    - **Revisions**: Subject to revisions for 3-4 months after initial release
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Role in Recession Dating")
        st.markdown("""
        The NBER Business Cycle Dating Committee considers Industrial Production as one of the
        **"Big Four"** recession indicators:

        1. Real Personal Income (less transfers)
        2. Nonfarm Payroll Employment
        3. Real Personal Consumption Expenditures
        4. **Industrial Production**

        According to the Fed's recession risk analysis, these indicators are combined into coincident
        economic indexes to identify business cycle turning points.
        """)

    with col2:
        st.subheader("How Investors Use Industrial Production")
        st.markdown("""
        1. **Economic Health Barometer**: Rising IP signals expansion; falling IP signals contraction
        2. **Capacity Utilization Context**: IP is released alongside capacity utilization
        3. **Sector Analysis**: Breakdown by industry provides manufacturing insights
        4. **Leading vs Coincident**: IP itself is coincident, but rate of change can provide warnings
        """)

    st.subheader("Academic Research on IP and Stock Returns")
    research = pd.DataFrame({
        'Finding': [
            'IP is coincident, not leading',
            'Stock returns lead IP',
            'Relationship unstable over time',
            'IP reacts quickly to business cycle',
            '"Big Four" indicator for recessions'
        ],
        'Source': ['NBER, Fed', 'International research', 'Stock & Watson (1998)',
                   'Chicago Fed', 'Advisor Perspectives'],
        'Implication': [
            'Cannot predict stock returns directly',
            'Markets anticipate production changes',
            'Historical patterns may not persist',
            'Useful for confirming regime changes',
            'Critical for identifying economic downturns'
        ]
    })
    st.dataframe(research, hide_index=True, use_container_width=True)

    st.subheader("Key Literature References")
    st.markdown("""
    - **Fama (1981)** - "Stock Returns, Real Activity, Inflation, and Money" established foundational
      work on the relationship between stock returns and real economic activity
    - **Balvers, Cosimano and McDonald (1990)** - Examined whether stock returns can be predicted
      by forecasts of industrial output
    - **Hong et al.** - Found that industry portfolios can lead the market by up to two months
    - **Stock and Watson (1998)** - Reported that the relationship between stock returns and
      production growth has not remained stable over time
    """)

    st.subheader("Limitations as a Stock Market Indicator")
    st.markdown("""
    1. **Coincident Nature**: By the time IP declines, stocks have often already priced in weakness
    2. **Sector Concentration**: Manufacturing is <20% of GDP; services dominate the modern economy
    3. **Data Revisions**: Initial releases are often revised, creating noise for trading decisions
    4. **Global Supply Chains**: U.S. production increasingly disconnected from U.S. corporate profits
    """)

elif analysis_id == 'xlre_orders_inv':
    st.header("Orders/Inventories Ratio")

    st.subheader("What is the Orders/Inventories Ratio?")
    st.markdown("""
    The Orders/Inventories Ratio measures the relationship between new manufacturing orders and
    existing inventory levels. Derived from the Census Bureau's Manufacturers' Shipments, Inventories,
    and Orders (M3) survey.

    **Key Characteristics:**
    - **Formula**: New Orders / Total Inventories
    - **Interpretation**: A rising ratio indicates demand is outpacing inventory (economic strength)
    - **Release Timing**: Published monthly by the Census Bureau
    - **Leading Property**: Changes in the ratio often precede broader economic shifts
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Signal Interpretation")
        signal_table = pd.DataFrame({
            'Ratio Direction': ['Rising', 'Falling', 'High Level', 'Low Level'],
            'Interpretation': [
                'Strong demand, lean inventories, economic expansion',
                'Weakening demand, inventory buildup, potential contraction',
                'Manufacturing capacity stretched, potential supply constraints',
                'Excess inventory, demand weakness'
            ]
        })
        st.dataframe(signal_table, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Role as a Leading Indicator")
        st.markdown("""
        According to Advisor Perspectives, the Orders/Inventories ratio is closely watched as part
        of the manufacturing sector's leading indicators:

        > *"Inventory levels are viewed as one of several key leading economic indicators. High
        inventory levels signal a degree of short-term and long-term market confidence."*

        The Conference Board's Leading Economic Index incorporates manufacturing orders data, noting
        that up to nine months in advance their index "does the best at signaling coming recessions."
        """)

    st.subheader("Why Real Estate is Sensitive to Manufacturing Indicators")

    sensitivity_reasons = pd.DataFrame({
        'Factor': ['Interest Rate Sensitivity', 'Economic Cycle Positioning', 'Leading Indicator Properties'],
        'Explanation': [
            'Manufacturing strength leads the Fed to maintain/raise rates, while weakness prompts rate cuts—directly impacting REIT valuations',
            'Demand for commercial real estate is closely tied to the macroeconomic cycle. Best REIT returns generated in early cycle.',
            'Listed REITs anticipate moves in direct real estate by 6-9 months. XLRE reacts quickly to economic signals.'
        ],
        'Source': ['Invesco Research', 'Nareit', 'Cohen & Steers']
    })
    st.dataframe(sensitivity_reasons, hide_index=True, use_container_width=True)

    st.subheader("Key Insights from Literature")
    insights = pd.DataFrame({
        'Finding': [
            'Manufacturing indicators lead economic cycle',
            'REITs outperform in rate-cutting environments',
            'Listed REITs lead direct real estate by 6-9 months',
            'Inventory buildup precedes recessions',
            'Best REIT returns in early cycle'
        ],
        'Source': ['Conference Board', 'Invesco', 'Cohen & Steers', 'Rosenberg Research', 'Nareit'],
        'Implication': [
            'O/I ratio provides early warning',
            'Falling O/I may signal rate cuts beneficial to REITs',
            'XLRE reacts quickly to economic signals',
            'Falling O/I ratio is bearish',
            'O/I regime helps identify cycle position'
        ]
    })
    st.dataframe(insights, hide_index=True, use_container_width=True)

    st.subheader("Limitations as a REIT Indicator")
    st.markdown("""
    1. **Sector Specificity**: Manufacturing is <20% of GDP; services dominate modern economy
    2. **Data Revisions**: Initial releases are often revised
    3. **Global Supply Chains**: U.S. manufacturing increasingly disconnected from domestic real estate demand
    4. **Short XLRE History**: XLRE launched in 2015, limiting historical validation
    5. **COVID Distortions**: 2020-2022 created unprecedented volatility
    """)

# Footer with references
st.divider()
st.caption("Sources: FRED, Census Bureau, Federal Reserve, NBER, academic literature as cited above.")

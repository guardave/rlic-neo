"""
Qualitative Analysis Page.

Displays indicator profiles, economic interpretation, and literature references.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.navigation import render_sidebar, render_breadcrumb, get_analysis_title

st.set_page_config(page_title="Qualitative | RLIC", page_icon="📖", layout="wide")

# Sidebar: Home, analysis selector, section links
analysis_id = render_sidebar(current_page="Qualitative")

# Content: Breadcrumb, then page
render_breadcrumb("Qualitative")
st.title(f"📖 Qualitative: {get_analysis_title()}")

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

elif analysis_id == 'xlp_retailirsa':
    st.header("XLP (Consumer Staples) vs Retail Inv/Sales")

    st.subheader("Sector Overview: Consumer Staples (XLP)")
    st.markdown("""
    XLP tracks companies producing **essential consumer products** that people buy regardless of
    economic conditions. Top holdings include Procter & Gamble, Coca-Cola, Costco, PepsiCo, and Walmart.

    **Key Characteristics:**
    - **Defensive Sector**: Low beta, stable demand across economic cycles
    - **Inelastic Demand**: Food, beverages, tobacco, household products are necessities
    - **Dividend Focus**: Many holdings are dividend aristocrats
    - **ETF Inception**: December 1998 (SPDR Select Sector)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Connection to Retail Inventories")
        st.markdown("""
        **Hypothesis**: XLP should show *weaker* sensitivity to RETAILIRSA because:

        1. **Inelastic Demand**: People buy necessities regardless of economic conditions
        2. **Stable Inventory Cycles**: Essential goods have more predictable demand patterns
        3. **Price Competition**: Private labels and staples pricing dampens margin volatility
        4. **Defensive Nature**: XLP often moves *opposite* to economic stress indicators
        """)

    with col2:
        st.subheader("Flight-to-Safety Dynamics")
        st.markdown("""
        During economic stress (when retail inventories rise):

        > *"Investors rotate from cyclical to defensive sectors during periods of uncertainty."*
        > - Fidelity Sector Research

        This means XLP may actually perform *better* when inventories are rising—not because
        rising inventories are good for staples, but because investors seek safety.
        """)

    st.subheader("Research Finding: Inverse Pattern")
    st.warning("""
    **Our Analysis Found**: XLP performs slightly *better* when retail inventories are rising
    (Sharpe 0.64 vs 0.55 for falling). This is NOT because rising inventories benefit staples—it's
    the **defensive rotation effect**. When economic stress builds (rising inventories signal
    weakening demand), investors rotate INTO defensive sectors like XLP.

    **Statistical Significance**: p = 0.785 — **NOT SIGNIFICANT**. The regime difference
    cannot be distinguished from random noise.
    """)

    st.subheader("Key Insights")
    insights = pd.DataFrame({
        'Finding': [
            'XLP is defensive, not cyclical',
            'Inverse pattern due to sector rotation',
            'Returns correlation near zero (-0.022)',
            'Regime difference not statistically significant',
            'SPY shows stronger RETAILIRSA relationship'
        ],
        'Implication': [
            'Do not expect direct economic sensitivity',
            'Rising inventories may benefit XLP via flight-to-safety',
            'RETAILIRSA cannot predict XLP returns',
            'Cannot recommend trading strategy based on this relationship',
            'Use broad market, not sector ETFs, for RETAILIRSA signals'
        ]
    })
    st.dataframe(insights, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown("""
    1. **Defensive Sector Dynamics**: XLP moves opposite to economic stress, confounding analysis
    2. **Sector Rotation Effects**: Flight-to-safety can mask underlying relationships
    3. **Short ETF History**: XLP launched in 1998, limiting historical validation
    4. **Aggregation Issues**: RETAILIRSA covers all retail, not just consumer staples
    5. **Statistical Insignificance**: p = 0.785 means the pattern could be random
    """)

elif analysis_id == 'xly_retailirsa':
    st.header("XLY (Consumer Discretionary) vs Retail Inv/Sales")

    st.subheader("Sector Overview: Consumer Discretionary (XLY)")
    st.markdown("""
    XLY tracks companies producing **non-essential consumer goods and services**. Top holdings
    include Amazon, Tesla, Home Depot, McDonald's, and Nike.

    **Key Characteristics:**
    - **Cyclical Sector**: High beta, performance tied to economic growth
    - **Elastic Demand**: Discretionary purchases expand/contract with consumer confidence
    - **Growth-Oriented**: Many holdings are growth stocks (Amazon, Tesla)
    - **ETF Inception**: December 1998 (SPDR Select Sector)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Connection to Retail Inventories")
        st.markdown("""
        **Hypothesis**: XLY should show *stronger* sensitivity to RETAILIRSA because:

        1. **Discretionary Demand is Elastic**: First to suffer when consumers cut back
        2. **Bullwhip Effect**: Discretionary items experience most pronounced inventory swings
        3. **Markup Sensitivity**: Excess inventory leads to markdowns on discretionary goods
        4. **Retailer Exposure**: XLY includes major retailers (Home Depot, Amazon retail)
        """)

    with col2:
        st.subheader("Consumer Confidence Link")
        st.markdown("""
        Consumer discretionary spending is directly tied to confidence and inventory dynamics:

        > *"Rising retail inventories signal weakening consumer demand—discretionary purchases
        are the first to be cut when consumers feel uncertain."* - Morningstar

        > *"Consumer discretionary is the most cyclical sector, showing the strongest correlation
        with the business cycle."* - Fidelity Sector Research
        """)

    st.subheader("Research Finding: Expected Pattern but Lacks Significance")
    st.info("""
    **Our Analysis Found**: XLY shows the economically expected pattern—performs better when
    retail inventories are falling (Sharpe 0.77 vs 0.31 for rising). The difference is
    substantial: +0.53% monthly (~6.4% annualized).

    **However**: p = 0.379 — **NOT STATISTICALLY SIGNIFICANT** at conventional thresholds.
    While the pattern matches economic intuition, the sample size and volatility prevent
    statistical confidence.
    """)

    st.subheader("Why XLY Doesn't Beat SPY")
    comparison = pd.DataFrame({
        'Metric': ['Returns Correlation', 'Regime Sharpe (Falling)', 'Regime Sharpe (Rising)', 'Regime Difference'],
        'XLY': ['-0.132', '0.77', '0.31', '+0.46'],
        'SPY': ['-0.105', '0.98', '0.52', '+0.46'],
        'Observation': [
            'XLY slightly stronger correlation',
            'SPY has better absolute performance',
            'SPY performs better even in "bad" regime',
            'Similar regime differentiation'
        ]
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("""
    **Why SPY is superior for RETAILIRSA analysis:**
    1. SPY has lower volatility, making patterns more detectable
    2. SPY diversification reduces idiosyncratic noise
    3. SPY has better absolute Sharpe ratios in both regimes
    """)

    st.subheader("Key Insights")
    insights = pd.DataFrame({
        'Finding': [
            'XLY shows expected economic pattern',
            'Performs better when inventories falling',
            'Pattern matches bullwhip effect theory',
            'Statistical significance not achieved',
            'Higher volatility requires more data'
        ],
        'Implication': [
            'Economic intuition is correct',
            'Falling inventories = strong consumer demand = good for XLY',
            'Discretionary items most sensitive to demand swings',
            'Cannot recommend trading strategy with confidence',
            'Need longer history or lower-vol instruments'
        ]
    })
    st.dataframe(insights, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown("""
    1. **Amazon Concentration**: Amazon is 25%+ of XLY, distorting sector dynamics
    2. **Statistical Insignificance**: p = 0.379 means pattern could be random
    3. **Higher Volatility**: XLY's ~6% monthly std requires more data for significance
    4. **Aggregation Issues**: RETAILIRSA covers all retail, not just discretionary
    5. **Structural Changes**: E-commerce has changed retail inventory dynamics
    """)

elif analysis_id == 'xlre_newhomesales':
    st.header("XLRE vs New Home Sales (HSN1F)")

    st.subheader("What is New Home Sales?")
    st.markdown("""
    **New Home Sales** (FRED: HSN1F) measures the number of newly constructed homes sold each month.
    Published monthly by the U.S. Census Bureau.

    **Key Characteristics:**
    - **Forward-Looking**: New home sales require mortgage applications, credit checks, and planning
    - **Construction Pipeline**: Sales drive future construction activity and employment
    - **Consumer Confidence**: Major financial decisions reflecting consumer outlook
    - **Release Timing**: ~3 weeks after the reference month
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Signal Interpretation")
        signal_table = pd.DataFrame({
            'Signal': ['Rising YoY', 'Falling YoY', 'Strong Positive', 'Strong Negative'],
            'Interpretation': [
                'Healthy housing demand, economic strength',
                'Weakening housing market, potential slowdown',
                'Construction boom ahead',
                'Recession warning signal'
            ]
        })
        st.dataframe(signal_table, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Why +8 Month Lead?")
        st.markdown("""
        **Our Analysis Found**: New Home Sales from 8 months ago predicts XLRE returns.

        **Why This Timing Makes Sense:**
        1. Home sales reflect buyer decisions from 3-6 months prior
        2. Construction activity follows sales by 6-12 months
        3. REIT revenue impacts show with multi-quarter delay
        4. Market may take time to fully price in housing trends
        """)

    st.subheader("Key Finding: Significant at Lag +8")
    st.success("""
    **POSITIVE RESULT**: r = 0.223, p = 0.025 at lag +8 months

    This means New Home Sales data from 8 months ago has predictive power for XLRE returns.
    The concurrent correlation (lag=0) is only r=0.06, demonstrating why **extended lead-lag
    analysis is critical** - the SOP fast-fail at lag=0 would have missed this relationship.
    """)

    st.subheader("Why XLRE, Not Individual REITs?")
    st.markdown("""
    **XLRE Composition:**
    - Dominated by commercial REITs (office, retail, industrial)
    - NOT homebuilders or residential construction
    - Still sensitive to housing market via:
      - Interest rate environment
      - Consumer wealth effects
      - Economic cycle positioning

    **Implication**: The relationship is indirect (housing → economy → REITs),
    which explains the 8-month lag rather than a more direct immediate effect.
    """)

    st.subheader("Trading Strategy Implication")
    strategy = pd.DataFrame({
        'Metric': ['Signal', 'Lag', 'Direction', 'Statistical Significance'],
        'Value': [
            'New Home Sales YoY change (from 8 months ago)',
            '+8 months',
            'Rising NHS → Positive XLRE returns expected',
            'p = 0.025 (significant at 5% level)'
        ]
    })
    st.dataframe(strategy, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown("""
    1. **Short XLRE History**: Only ~109 months of data (XLRE launched October 2015)
    2. **Indirect Relationship**: Housing affects REITs through multiple channels
    3. **COVID Distortions**: 2020-2022 created unprecedented housing volatility
    4. **Composition Mismatch**: XLRE is commercial REITs, not residential builders
    5. **Single Significant Lag**: Only lag +8 is significant; others are marginal
    """)

elif analysis_id == 'xli_ism_mfg':
    st.header("XLI vs ISM Manufacturing PMI (NAPM)")

    st.subheader("What is ISM Manufacturing PMI?")
    st.markdown("""
    The **ISM Manufacturing PMI** (Purchasing Managers' Index) is a monthly survey of 300+ manufacturing
    purchasing managers conducted by the Institute for Supply Management. Published on the **1st business
    day** of each month, making it one of the earliest economic releases.

    **Key Characteristics:**
    - **Diffusion Index**: PMI > 50 = expansion, PMI = 50 = no change, PMI < 50 = contraction
    - **Sub-components**: New Orders, Production, Employment, Supplier Deliveries, Inventories
    - **Leading Property**: Survey data captures real-time sentiment before hard production data
    - **History**: Available since 1948, one of the longest-running economic surveys
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Economic Signal Interpretation")
        signal_table = pd.DataFrame({
            'PMI Level': ['Above 50', 'Below 50', 'Above 55', 'Below 45'],
            'Interpretation': [
                'Manufacturing sector expanding',
                'Manufacturing sector contracting',
                'Strong expansion, capacity pressure building',
                'Severe contraction, recession risk elevated'
            ]
        })
        st.dataframe(signal_table, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Why XLI (Industrials)?")
        st.markdown("""
        **XLI Composition:** Boeing, Caterpillar, Honeywell, GE Aerospace, Union Pacific,
        Lockheed Martin, RTX, Deere & Co.

        **Direct Link**: XLI holdings ARE the manufacturers being surveyed by ISM.
        When purchasing managers report expansion, XLI companies are seeing stronger orders.

        **Timing Advantage**: PMI is released before industrial production data (INDPRO),
        giving an earlier signal about manufacturing health.
        """)

    st.subheader("Key Research Finding: Confirmatory, Not Predictive")
    st.warning("""
    **Our Analysis Found**: ISM Manufacturing PMI does NOT predict XLI returns. All significant
    correlations are at **negative lags** (best: lag -4, r=0.241, p<0.0001), meaning the stock
    market moves FIRST and ISM PMI confirms later.

    **However**, the PMI > 50 regime segmentation IS meaningful:
    - **Expansion** (PMI > 50): Mean return +1.34%/mo, Sharpe 0.93
    - **Contraction** (PMI <= 50): Mean return -0.22%/mo, Sharpe -0.12
    - Regime difference: p = 0.019 (statistically significant)

    This means ISM PMI is useful as a **contemporaneous regime indicator** but cannot be used
    as a leading signal for trading.
    """)

    st.subheader("Academic and Professional Research")
    research = pd.DataFrame({
        'Finding': [
            'ISM PMI leads industrial production by 1-2 months',
            'Stock markets are forward-looking and price in PMI changes early',
            'PMI above 50 correlates with GDP growth',
            'New Orders sub-index is most predictive component',
            'ISM PMI is published before most hard economic data'
        ],
        'Source': ['Federal Reserve', 'Fama (1981)', 'ISM Research', 'Stock & Watson', 'BLS Calendar'],
        'Implication': [
            'PMI leads real economy but not equities',
            'Equities anticipate PMI rather than react to it',
            'Strong macro signal for economic cycle positioning',
            'Sub-components may add value beyond headline PMI',
            'Information advantage vs hard data but not vs markets'
        ]
    })
    st.dataframe(research, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown("""
    1. **Not Predictive for Equities**: Stock market moves before PMI data is released
    2. **Survey Bias**: Self-reported data from purchasing managers, not objective measurement
    3. **Diffusion vs Magnitude**: PMI measures breadth (% reporting growth) not magnitude
    4. **Manufacturing Focus**: Only ~12% of GDP is manufacturing; services dominate
    5. **Market Pricing**: PMI is closely watched; moves may already be priced in
    """)

elif analysis_id == 'xli_ism_svc':
    st.header("XLI vs ISM Services PMI (Non-Manufacturing)")

    st.subheader("What is ISM Services PMI?")
    st.markdown("""
    The **ISM Services PMI** (also known as Non-Manufacturing PMI) is a monthly survey of purchasing
    managers at services-sector companies. Published by the Institute for Supply Management on the
    **3rd business day of each month** (2 days after Manufacturing PMI).

    **Key characteristics:**
    - Covers **~80% of GDP** (services dominate the US economy)
    - Surveys sectors like healthcare, finance, retail, transportation, professional services
    - Composite index from 4 subindices: Business Activity, New Orders, Employment, Supplier Deliveries
    - **PMI > 50** = services sector expansion; **PMI ≤ 50** = contraction
    """)

    st.subheader("Economic Signal Interpretation")
    interpretation_data = {
        'PMI Range': ['> 55', '50-55', '< 50', '< 45'],
        'Signal': ['Strong Expansion', 'Moderate Growth', 'Contraction', 'Severe Contraction'],
        'Economic Meaning': [
            'Services sector growing robustly, broad-based expansion',
            'Growth continuing but at a slower pace',
            'Services sector shrinking, potential recession risk',
            'Deep contraction, typically recession territory'
        ]
    }
    st.table(pd.DataFrame(interpretation_data))

    st.subheader("Why Compare with XLI (Industrials)?")
    st.markdown("""
    **Hypothesis**: Services PMI may influence Industrials because:
    1. **Input-output linkages**: Services firms purchase manufactured goods
    2. **GDP proxy**: Services = ~80% of GDP, so broad health affects all sectors
    3. **Employment signal**: Services employment trends affect consumer spending on industrial goods

    **Counter-hypothesis**: The relationship may be weak because:
    - XLI holds **manufacturing** companies, not services
    - Manufacturing PMI is a more direct indicator for XLI
    - Services PMI may be better suited for consumer/financial sector ETFs
    """)

    st.subheader("Key Finding")
    st.warning("""
    ⚠️ **CONFIRMATORY, NOT PREDICTIVE**: ISM Services PMI does NOT predict XLI returns.

    All 8 significant lags are **negative** (lag -1 through -11), meaning XLI returns
    move FIRST, then Services PMI follows 1-11 months later.

    **Best lag: -1 month** (r=0.317, p<0.0001) — XLI leads Services PMI by 1 month.
    """)

    st.markdown("""
    **Despite reverse causality, regime filtering is useful:**
    - **Svc Expansion** (PMI > 50): Mean +1.05%/mo, Sharpe 0.79
    - **Svc Contraction** (PMI ≤ 50): Mean -3.94%/mo, Sharpe -1.38

    The regime difference is highly significant (p<0.0001).
    """)

    st.subheader("Data Limitation")
    st.info("""
    📊 **Data Period**: December 1999 to April 2020 (245 months)

    The ISM Services data was assembled from historical sources (forecasts.org) and
    hardcoded ISM press release values. Recent data (2020-05+) is not available because
    FRED discontinued the NMFBAI series and alternative web sources returned errors.
    """)

    st.subheader("Comparison with Manufacturing PMI")
    comparison_data = {
        'Metric': ['Best Lag', 'Best r', 'Significant Lags', 'Direction', 'Expansion Sharpe', 'Contraction Sharpe', 'Observations'],
        'Manufacturing': ['-4', '0.241', '11', 'All negative', '0.93', '-0.12', '314'],
        'Services': ['-1', '0.317', '8', 'All negative', '0.79', '-1.38', '245']
    }
    st.table(pd.DataFrame(comparison_data))

    st.markdown("""
    **Key differences:**
    - Services PMI shows **stronger concurrent correlation** (r=0.317 at lag -1 vs r=0.241 at lag -4)
    - Both are confirmatory (all significant lags negative)
    - Services has **more extreme contraction penalty** (Sharpe -1.38 vs -0.12)
    - Manufacturing has more data (314 vs 245 observations)
    """)

else:
    # Fallback for unknown analysis types
    st.warning(f"Qualitative content for '{analysis_id}' has not been created yet.")
    st.markdown("""
    This analysis is available in the dashboard but detailed qualitative content
    has not yet been added. Please refer to the quantitative tabs (Correlation,
    Lead-Lag, Regimes) for data-driven insights.
    """)

# Footer with references
st.divider()
st.caption("Sources: FRED, Census Bureau, Federal Reserve, NBER, Fidelity, Morningstar, academic literature as cited above.")

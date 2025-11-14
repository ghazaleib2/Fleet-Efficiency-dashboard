"""
streamlit run fleet_dashboard.py
Fleet Efficiency Analysis Dashboard v1.0
========================================
Professional online dashboard for WAY S.r.l. fleet efficiency monitoring
Built for executive presentation and academic reporting

Features:
- Real-time KPI monitoring across 697 vehicles and 69 fleets
- Category-aware benchmarking (LCV, MDV, HDV, Passenger Cars)
- Interactive fleet and vehicle-level analytics
- Anomaly detection and data quality diagnostics
- Multi-dimensional comparisons by fuel type, brand, and category

Design Philosophy:
- Executive-ready visualizations with clear insights
- Category-specific benchmarks to avoid false positives
- Comprehensive data quality transparency
- Sustainability-focused metrics (CO₂, fuel efficiency)

Deployment: Streamlit Cloud, Render, or any Python hosting service
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fleet Efficiency Dashboard | WAY S.r.l.",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #2ca02c;
        --accent-color: #ff7f0e;
        --background-dark: #0e1117;
        --card-background: #262730;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    /* Headers */
    .dashboard-header {
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .section-header {
        color: #1f77b4;
        border-bottom: 3px solid #2ca02c;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ff7f0e;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9em;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load all CSV datasets with error handling"""
    try:
        data = {
            'daily': pd.read_csv('fleet_efficiency_daily.csv'),
            'vehicle': pd.read_csv('efficiency_by_vehicle.csv'),
            'fleet': pd.read_csv('efficiency_by_fleet.csv'),
            'category': pd.read_csv('efficiency_by_category.csv'),
            'brand': pd.read_csv('efficiency_by_brand_model.csv'),
            'anomalies': pd.read_csv('anomalies_category_aware.csv'),
            'excluded': pd.read_csv('excluded_records.csv')
        }
        
        # ===================================================================
        # NORMALIZE COLUMNS TO HANDLE SNAKE_CASE AND CAMELCASE VARIATIONS
        # ===================================================================
        
        def _rename(df, mapping):
            """Rename columns if they exist in the dataframe"""
            exist = {k: v for k, v in mapping.items() if k in df.columns}
            return df.rename(columns=exist)
        
        def _lower(df):
            """Normalize column names to lowercase for robust matching"""
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        
        # Normalize all dataframes to lowercase first
        for key in list(data.keys()):
            data[key] = _lower(data[key])
        
        # VEHICLE file mappings (snake_case → CamelCase)
        data['vehicle'] = _rename(data['vehicle'], {
            'vehicle_name': 'Vehicle',
            'vehicle': 'Vehicle',
            'fleet_name': 'Fleet',
            'fleet': 'Fleet',
            'category': 'Category',
            'fuel_type': 'Fuel_Type',
            'brand_model': 'Brand_Model',
            'brand': 'Brand_Model',
            'distance_traveled_km': 'Total_Distance_km',
            'total_distance_km': 'Total_Distance_km',
            'liters_consumed': 'Total_Fuel_L',
            'total_fuel_l': 'Total_Fuel_L',
            'km_per_liter': 'Efficiency_km_per_L',
            'efficiency_km_per_l': 'Efficiency_km_per_L',
            'total_co2_kg': 'Total_CO2_kg',
            'total_fuel_cost_eur': 'Total_Cost_EUR',
            'cost_per_km_eur': 'Cost_per_km_EUR'
        })
        
        # FLEET file mappings
        data['fleet'] = _rename(data['fleet'], {
            'fleet': 'Fleet',
            'fleet_name': 'Fleet',
            'vehicle_count': 'Vehicle_Count',
            'distance_traveled_km': 'Total_Distance_km',
            'total_distance_km': 'Total_Distance_km',
            'liters_consumed': 'Total_Fuel_L',
            'total_fuel_l': 'Total_Fuel_L',
            'km_per_liter': 'Avg_Efficiency_km_per_L',
            'avg_efficiency_km_per_l': 'Avg_Efficiency_km_per_L',
            'total_co2_kg': 'Total_CO2_kg',
            'total_fuel_cost_eur': 'Total_Cost_EUR'
        })
        
        # CATEGORY file mappings
        data['category'] = _rename(data['category'], {
            'category': 'Category',
            'vehicle_count': 'Vehicle_Count',
            'distance_traveled_km': 'Total_Distance_km',
            'total_distance_km': 'Total_Distance_km',
            'liters_consumed': 'Total_Fuel_L',
            'total_fuel_l': 'Total_Fuel_L',
            'km_per_liter': 'Avg_Efficiency_km_per_L',
            'avg_efficiency_km_per_l': 'Avg_Efficiency_km_per_L',
            'avg_idle_ratio': 'Avg_Idle_Ratio',
            'avg_cost_per_km_eur': 'Avg_Cost_per_km_EUR',
            'avg_co2_per_km_kg': 'Avg_CO2_per_km_kg',
            'total_co2_kg': 'Total_CO2_kg',
            'total_fuel_cost_eur': 'Total_Cost_EUR'
        })
        
        # BRAND file mappings
        data['brand'] = _rename(data['brand'], {
            'brand_model': 'Brand_Model',
            'brand': 'Brand_Model',
            'vehicle_count': 'Vehicle_Count',
            'distance_traveled_km': 'Total_Distance_km',
            'total_distance_km': 'Total_Distance_km',
            'liters_consumed': 'Total_Fuel_L',
            'total_fuel_l': 'Total_Fuel_L',
            'km_per_liter': 'Avg_Efficiency_km_per_L',
            'avg_efficiency_km_per_l': 'Avg_Efficiency_km_per_L',
            'total_co2_kg': 'Total_CO2_kg',
            'total_fuel_cost_eur': 'Total_Cost_EUR'
        })
        
        # DAILY file mappings
        data['daily'] = _rename(data['daily'], {
            'date': 'Date',
            'vehicle_name': 'Vehicle',
            'vehicle': 'Vehicle',
            'fleet_name': 'Fleet',
            'fleet': 'Fleet',
            'distance_traveled_km': 'Distance_km',
            'distance_km': 'Distance_km',
            'liters_consumed': 'Fuel_L',
            'fuel_l': 'Fuel_L',
            'km_per_liter': 'Efficiency_km_per_L',
            'efficiency_km_per_l': 'Efficiency_km_per_L',
            'co2_kg': 'CO2_kg',
            'total_co2_kg': 'CO2_kg'
        })
        
        # ANOMALIES file mappings
        data['anomalies'] = _rename(data['anomalies'], {
            'date': 'Date',
            'vehicle_name': 'Vehicle',
            'vehicle': 'Vehicle',
            'fleet_name': 'Fleet',
            'fleet': 'Fleet',
            'category': 'Category',
            'anomaly_types': 'Anomaly_Type',
            'anomaly_type': 'Anomaly_Type',
            'km_per_liter': 'Efficiency_km_per_L',
            'efficiency_km_per_l': 'Efficiency_km_per_L',
            'distance_km': 'Distance_km',
            'distance_traveled_km': 'Distance_km',
            'idle_ratio': 'Idle_Ratio'
        })
        
        # EXCLUDED file mappings
        data['excluded'] = _rename(data['excluded'], {
            'date': 'Date',
            'vehicle_name': 'Vehicle',
            'vehicle': 'Vehicle',
            'fleet_name': 'Fleet',
            'fleet': 'Fleet',
            'distance_traveled_km': 'Distance_km',
            'distance_km': 'Distance_km',
            'liters_consumed': 'Fuel_L',
            'fuel_l': 'Fuel_L',
            'km_per_liter': 'Efficiency_km_per_L',
            'efficiency_km_per_l': 'Efficiency_km_per_L',
            'exclusion_reason': 'Exclusion_Reason'
        })
        
        # ===================================================================
        # Convert date columns after normalization
        # ===================================================================
        if 'Date' in data['daily'].columns:
            data['daily']['Date'] = pd.to_datetime(data['daily']['Date'], errors='coerce')
        if 'Date' in data['anomalies'].columns:
            data['anomalies']['Date'] = pd.to_datetime(data['anomalies']['Date'], errors='coerce')
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all CSV files are in the same directory as this script.")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_number(num, decimals=2, suffix=""):
    """Format numbers with thousands separator"""
    if pd.isna(num):
        return "N/A"
    if suffix == "€":
        return f"€{num:,.{decimals}f}"
    elif suffix == "%":
        return f"{num:.{decimals}f}%"
    return f"{num:,.{decimals}f}{suffix}"

def create_kpi_card(label, value, icon="📊"):
    """Create a styled KPI metric card"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 2em;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def get_category_color(category):
    """Assign consistent colors to vehicle categories"""
    colors = {
        'Passenger Cars': '#1f77b4',
        'Light Commercial Vehicle': '#ff7f0e',
        'Medium Duty Vehicle': '#2ca02c',
        'Heavy Duty Vehicle': '#d62728',
        'Default': '#9467bd'
    }
    return colors.get(category, '#7f7f7f')

def get_benchmark(category):
    """Get category-specific efficiency benchmarks"""
    benchmarks = {
        'Passenger Cars': 12.0,
        'Light Commercial Vehicle': 9.0,
        'Medium Duty Vehicle': 6.0,
        'Heavy Duty Vehicle': 4.5,
        'Default': 8.0
    }
    return benchmarks.get(category, 8.0)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_efficiency_gauge(value, benchmark, title="Fleet Efficiency"):
    """Create a gauge chart for efficiency metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': benchmark, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, benchmark * 1.5], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, benchmark * 0.7], 'color': '#ffcccc'},
                {'range': [benchmark * 0.7, benchmark * 0.9], 'color': '#ffffcc'},
                {'range': [benchmark * 0.9, benchmark * 1.5], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': benchmark
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': "Arial"}
    )
    return fig

def create_trend_chart(df, x_col, y_col, title, color='#1f77b4'):
    """Create an area trend chart"""
    fig = px.area(df, x=x_col, y=y_col, title=title)
    fig.update_traces(
        fill='tozeroy',
        line_color=color,
        fillcolor=f'rgba(31, 119, 180, 0.3)'
    )
    fig.update_layout(
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    return fig

def create_comparison_radar(df, categories, metrics, title):
    """Create a radar chart for multi-metric comparison"""
    fig = go.Figure()
    
    for category in categories:
        cat_data = df[df['Category'] == category] if 'Category' in df.columns else df
        if len(cat_data) > 0:
            values = [cat_data[metric].values[0] if len(cat_data) > 0 else 0 for metric in metrics]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=category,
                line_color=get_category_color(category)
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max([100, df[metrics].max().max()])]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title=title,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Load data
    data = load_data()
    
    if data is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("## 🚛 Fleet Dashboard")
    st.sidebar.markdown("**WAY S.r.l. | Fleet Efficiency System**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🏢 Fleet Comparison", "📂 Category Insights", 
         "🚗 Vehicle Explorer", "⚠️ Anomaly Dashboard", "🔍 Data Quality"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    total_vehicles = len(data['vehicle'])
    total_fleets = len(data['fleet'])
    avg_efficiency = data['vehicle']['Efficiency_km_per_L'].mean()
    
    st.sidebar.metric("Total Vehicles", f"{total_vehicles:,}")
    st.sidebar.metric("Total Fleets", f"{total_fleets:,}")
    st.sidebar.metric("Avg Efficiency", f"{avg_efficiency:.2f} km/L")
    
    st.sidebar.markdown("---")
    
    # Debug: Show column names (expandable)
    with st.sidebar.expander("🔍 Debug: Column Names"):
        st.write("**Vehicle columns:**")
        st.code(", ".join(data['vehicle'].columns.tolist()))
        st.write("**Fleet columns:**")
        st.code(", ".join(data['fleet'].columns.tolist()))
        st.write("**Category columns:**")
        st.code(", ".join(data['category'].columns.tolist()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("**v5.6** | Politecnico di Torino × WAY S.r.l.")
    
    # ========================================================================
    # PAGE: OVERVIEW
    # ========================================================================
    
    if page == "📊 Overview":
        st.markdown('<div class="dashboard-header"><h1>🚛 Fleet Efficiency Dashboard</h1><p>Comprehensive Analytics for WAY S.r.l. Vehicle Fleet</p></div>', unsafe_allow_html=True)
        
        # Top KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_km = data['vehicle']['Total_Distance_km'].sum()
        total_fuel = data['vehicle']['Total_Fuel_L'].sum()
        total_co2 = data['vehicle']['Total_CO2_kg'].sum()
        total_cost = data['vehicle']['Total_Cost_EUR'].sum()
        avg_eff = data['vehicle']['Efficiency_km_per_L'].mean()
        
        with col1:
            st.markdown(create_kpi_card("Total Distance", format_number(total_km, 0, " km"), "🛣️"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_kpi_card("Total Fuel", format_number(total_fuel, 0, " L"), "⛽"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_kpi_card("Total CO₂", format_number(total_co2/1000, 1, " tons"), "🌍"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_kpi_card("Total Cost", format_number(total_cost, 0, "€"), "💰"), unsafe_allow_html=True)
        with col5:
            st.markdown(create_kpi_card("Avg Efficiency", format_number(avg_eff, 2, " km/L"), "📊"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Efficiency Overview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 Fleet Efficiency vs Benchmark")
            category_eff = data['category'].copy()
            category_eff['Benchmark'] = category_eff['Category'].apply(get_benchmark)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Actual Efficiency',
                x=category_eff['Category'],
                y=category_eff['Avg_Efficiency_km_per_L'],
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                name='Benchmark',
                x=category_eff['Category'],
                y=category_eff['Benchmark'],
                marker_color='#2ca02c',
                opacity=0.6
            ))
            fig.update_layout(
                barmode='group',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(title='km/L', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🌍 CO₂ Emissions by Category")
            fig = px.pie(
                data['category'],
                values='Total_CO2_kg',
                names='Category',
                color='Category',
                color_discrete_map={cat: get_category_color(cat) for cat in data['category']['Category'].unique()}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=400,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Daily Trends
        st.markdown("### 📈 Daily Performance Trends")
        
        if 'Date' in data['daily'].columns and len(data['daily']) > 0:
            daily_agg = data['daily'].groupby('Date').agg({
                'Distance_km': 'sum',
                'Fuel_L': 'sum',
                'CO2_kg': 'sum',
                'Efficiency_km_per_L': 'mean'
            }).reset_index()
            
            tab1, tab2, tab3 = st.tabs(["📊 Efficiency", "⛽ Fuel Consumption", "🌍 CO₂ Emissions"])
            
            with tab1:
                fig = create_trend_chart(daily_agg, 'Date', 'Efficiency_km_per_L', 'Daily Average Efficiency', '#1f77b4')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = create_trend_chart(daily_agg, 'Date', 'Fuel_L', 'Daily Fuel Consumption (L)', '#ff7f0e')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = create_trend_chart(daily_agg, 'Date', 'CO2_kg', 'Daily CO₂ Emissions (kg)', '#2ca02c')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Daily trend data not available or missing Date column")
        
        st.markdown("---")
        
        # Fleet Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📂 Vehicle Count by Category")
            category_counts = data['category'].copy()
            fig = px.bar(
                category_counts.sort_values('Vehicle_Count', ascending=False),
                x='Category',
                y='Vehicle_Count',
                color='Category',
                color_discrete_map={cat: get_category_color(cat) for cat in category_counts['Category'].unique()},
                text='Vehicle_Count'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(title='', showgrid=False),
                yaxis=dict(title='Number of Vehicles', showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏭 Top 10 Brands by Fleet Size")
            brand_counts = data['brand'].nlargest(10, 'Vehicle_Count')
            fig = px.bar(
                brand_counts,
                y='Brand_Model',
                x='Vehicle_Count',
                orientation='h',
                color='Vehicle_Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(title='Number of Vehicles', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='', showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE: FLEET COMPARISON
    # ========================================================================
    
    elif page == "🏢 Fleet Comparison":
        st.markdown("# 🏢 Fleet-Level Comparison")
        st.markdown("Compare performance across 69 organizational fleets")
        
        # Fleet Rankings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top 10 Most Efficient Fleets")
            top_fleets = data['fleet'].nlargest(10, 'Avg_Efficiency_km_per_L')[['Fleet', 'Avg_Efficiency_km_per_L', 'Vehicle_Count']]
            top_fleets['Rank'] = range(1, 11)
            top_fleets = top_fleets[['Rank', 'Fleet', 'Avg_Efficiency_km_per_L', 'Vehicle_Count']]
            top_fleets.columns = ['Rank', 'Fleet', 'Efficiency (km/L)', 'Vehicles']
            st.dataframe(
                top_fleets.style.format({'Efficiency (km/L)': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("### ⚠️ Bottom 10 Fleets (Need Attention)")
            bottom_fleets = data['fleet'].nsmallest(10, 'Avg_Efficiency_km_per_L')[['Fleet', 'Avg_Efficiency_km_per_L', 'Vehicle_Count']]
            bottom_fleets['Rank'] = range(1, 11)
            bottom_fleets = bottom_fleets[['Rank', 'Fleet', 'Avg_Efficiency_km_per_L', 'Vehicle_Count']]
            bottom_fleets.columns = ['Rank', 'Fleet', 'Efficiency (km/L)', 'Vehicles']
            st.dataframe(
                bottom_fleets.style.format({'Efficiency (km/L)': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # Fleet Distribution
        st.markdown("### 📊 Fleet Efficiency Distribution")
        
        fig = px.histogram(
            data['fleet'],
            x='Avg_Efficiency_km_per_L',
            nbins=30,
            title='Distribution of Fleet Average Efficiency',
            labels={'Avg_Efficiency_km_per_L': 'Efficiency (km/L)', 'count': 'Number of Fleets'}
        )
        fig.add_vline(
            x=data['fleet']['Avg_Efficiency_km_per_L'].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {data['fleet']['Avg_Efficiency_km_per_L'].median():.2f} km/L"
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Cost vs Efficiency Analysis
        st.markdown("### 💰 Cost vs Efficiency Analysis")
        
        fig = px.scatter(
            data['fleet'],
            x='Avg_Efficiency_km_per_L',
            y='Total_Cost_EUR',
            size='Vehicle_Count',
            hover_data=['Fleet'],
            title='Fleet Cost vs Efficiency (bubble size = vehicle count)',
            labels={'Avg_Efficiency_km_per_L': 'Efficiency (km/L)', 'Total_Cost_EUR': 'Total Cost (€)'}
        )
        fig.update_traces(marker=dict(color='#1f77b4', line=dict(width=1, color='white')))
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Fleet Details Table
        st.markdown("### 📋 Complete Fleet Summary")
        
        fleet_summary = data['fleet'][['Fleet', 'Vehicle_Count', 'Total_Distance_km', 'Total_Fuel_L', 
                                        'Avg_Efficiency_km_per_L', 'Total_Cost_EUR', 'Total_CO2_kg']].copy()
        fleet_summary = fleet_summary.sort_values('Avg_Efficiency_km_per_L', ascending=False)
        fleet_summary['Total_CO2_tons'] = fleet_summary['Total_CO2_kg'] / 1000
        fleet_summary = fleet_summary.drop('Total_CO2_kg', axis=1)
        
        st.dataframe(
            fleet_summary.style.format({
                'Total_Distance_km': '{:,.0f}',
                'Total_Fuel_L': '{:,.0f}',
                'Avg_Efficiency_km_per_L': '{:.2f}',
                'Total_Cost_EUR': '€{:,.2f}',
                'Total_CO2_tons': '{:,.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    # ========================================================================
    # PAGE: CATEGORY INSIGHTS
    # ========================================================================
    
    elif page == "📂 Category Insights":
        st.markdown("# 📂 Category-Specific Analysis")
        st.markdown("Deep dive into vehicle categories with category-aware benchmarks")
        
        # Category Overview Cards
        st.markdown("### 📊 Category Performance Summary")
        
        categories = data['category'].sort_values('Avg_Efficiency_km_per_L', ascending=False)
        
        for idx, row in categories.iterrows():
            with st.expander(f"**{row['Category']}** — {row['Vehicle_Count']} vehicles | {row['Avg_Efficiency_km_per_L']:.2f} km/L", expanded=(idx==0)):
                col1, col2, col3, col4 = st.columns(4)
                
                benchmark = get_benchmark(row['Category'])
                efficiency_pct = ((row['Avg_Efficiency_km_per_L'] / benchmark) - 1) * 100
                
                with col1:
                    st.metric("Avg Efficiency", f"{row['Avg_Efficiency_km_per_L']:.2f} km/L", 
                             f"{efficiency_pct:+.1f}% vs benchmark")
                with col2:
                    st.metric("Total Distance", f"{row['Total_Distance_km']:,.0f} km")
                with col3:
                    st.metric("Total Fuel", f"{row['Total_Fuel_L']:,.0f} L")
                with col4:
                    st.metric("CO₂ Emissions", f"{row['Total_CO2_kg']/1000:.1f} tons")
                
                # Category-specific metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Avg_Idle_Ratio' in row.index:
                        idle_pct = row['Avg_Idle_Ratio'] * 100
                        st.metric("Avg Idle Time", f"{idle_pct:.1f}%")
                    if 'Avg_Cost_per_km_EUR' in row.index:
                        st.metric("Cost per km", f"€{row['Avg_Cost_per_km_EUR']:.3f}")
                
                with col2:
                    if 'Avg_CO2_per_km_kg' in row.index:
                        st.metric("CO₂ per km", f"{row['Avg_CO2_per_km_kg']:.3f} kg")
                    st.metric("Benchmark", f"{benchmark:.2f} km/L", delta_color="off")
        
        st.markdown("---")
        
        # Comparative Analysis
        st.markdown("### 🔄 Cross-Category Comparison")
        
        tab1, tab2, tab3 = st.tabs(["📊 Efficiency", "💰 Cost Analysis", "🌍 Environmental Impact"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                categories_sorted = data['category'].sort_values('Avg_Efficiency_km_per_L')
                fig.add_trace(go.Bar(
                    y=categories_sorted['Category'],
                    x=categories_sorted['Avg_Efficiency_km_per_L'],
                    orientation='h',
                    marker=dict(
                        color=[get_category_color(cat) for cat in categories_sorted['Category']],
                        line=dict(color='white', width=1)
                    ),
                    text=categories_sorted['Avg_Efficiency_km_per_L'].round(2),
                    textposition='auto'
                ))
                fig.update_layout(
                    title='Average Efficiency by Category',
                    xaxis_title='km/L',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Benchmark comparison
                cat_bench = data['category'].copy()
                cat_bench['Benchmark'] = cat_bench['Category'].apply(get_benchmark)
                cat_bench['Performance_%'] = ((cat_bench['Avg_Efficiency_km_per_L'] / cat_bench['Benchmark']) - 1) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=cat_bench['Category'],
                    x=cat_bench['Performance_%'],
                    orientation='h',
                    marker=dict(
                        color=cat_bench['Performance_%'],
                        colorscale='RdYlGn',
                        line=dict(color='white', width=1),
                        cmin=-20,
                        cmax=20
                    ),
                    text=cat_bench['Performance_%'].round(1),
                    texttemplate='%{text:+.1f}%',
                    textposition='auto'
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                fig.update_layout(
                    title='Performance vs Category Benchmark',
                    xaxis_title='% Deviation from Benchmark',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    data['category'].sort_values('Total_Cost_EUR', ascending=False),
                    x='Category',
                    y='Total_Cost_EUR',
                    color='Category',
                    color_discrete_map={cat: get_category_color(cat) for cat in data['category']['Category'].unique()},
                    title='Total Operating Cost by Category'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis_title='Total Cost (€)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Avg_Cost_per_km_EUR' in data['category'].columns:
                    fig = px.bar(
                        data['category'].sort_values('Avg_Cost_per_km_EUR', ascending=False),
                        x='Category',
                        y='Avg_Cost_per_km_EUR',
                        color='Category',
                        color_discrete_map={cat: get_category_color(cat) for cat in data['category']['Category'].unique()},
                        title='Cost per Kilometer by Category'
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title='Cost per km (€)',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    data['category'],
                    values='Total_CO2_kg',
                    names='Category',
                    title='CO₂ Contribution by Category',
                    color='Category',
                    color_discrete_map={cat: get_category_color(cat) for cat in data['category']['Category'].unique()}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Avg_CO2_per_km_kg' in data['category'].columns:
                    fig = px.bar(
                        data['category'].sort_values('Avg_CO2_per_km_kg', ascending=False),
                        x='Category',
                        y='Avg_CO2_per_km_kg',
                        color='Category',
                        color_discrete_map={cat: get_category_color(cat) for cat in data['category']['Category'].unique()},
                        title='CO₂ Intensity (per km)'
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title='CO₂ per km (kg)',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE: VEHICLE EXPLORER
    # ========================================================================
    
    elif page == "🚗 Vehicle Explorer":
        st.markdown("# 🚗 Vehicle-Level Explorer")
        st.markdown("Detailed analysis of individual vehicles")
        
        # Filters
        st.markdown("### 🔍 Filter Vehicles")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            categories = ['All'] + list(data['vehicle']['Category'].unique()) if 'Category' in data['vehicle'].columns else ['All']
            selected_category = st.selectbox("Category", categories)
        
        with col2:
            fleets = ['All'] + sorted(data['vehicle']['Fleet'].unique()) if 'Fleet' in data['vehicle'].columns else ['All']
            selected_fleet = st.selectbox("Fleet", fleets)
        
        with col3:
            fuel_types = ['All'] + list(data['vehicle']['Fuel_Type'].unique()) if 'Fuel_Type' in data['vehicle'].columns else ['All']
            selected_fuel = st.selectbox("Fuel Type", fuel_types)
        
        with col4:
            search_term = st.text_input("Search Vehicle Name", "")
        
        # Apply filters
        filtered_vehicles = data['vehicle'].copy()
        
        if selected_category != 'All' and 'Category' in filtered_vehicles.columns:
            filtered_vehicles = filtered_vehicles[filtered_vehicles['Category'] == selected_category]
        if selected_fleet != 'All' and 'Fleet' in filtered_vehicles.columns:
            filtered_vehicles = filtered_vehicles[filtered_vehicles['Fleet'] == selected_fleet]
        if selected_fuel != 'All' and 'Fuel_Type' in filtered_vehicles.columns:
            filtered_vehicles = filtered_vehicles[filtered_vehicles['Fuel_Type'] == selected_fuel]
        if search_term and 'Vehicle' in filtered_vehicles.columns:
            filtered_vehicles = filtered_vehicles[filtered_vehicles['Vehicle'].str.contains(search_term, case=False, na=False)]
        
        st.markdown(f"**{len(filtered_vehicles):,} vehicles** match your filters")
        
        st.markdown("---")
        
        # Vehicle Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Efficiency Distribution (Filtered)")
            fig = px.histogram(
                filtered_vehicles,
                x='Efficiency_km_per_L',
                nbins=40,
                title='Vehicle Efficiency Distribution'
            )
            fig.add_vline(
                x=filtered_vehicles['Efficiency_km_per_L'].median(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {filtered_vehicles['Efficiency_km_per_L'].median():.2f}"
            )
            fig.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(title='km/L', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='Count', showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Cost per km Distribution")
            if 'Cost_per_km_EUR' in filtered_vehicles.columns:
                fig = px.histogram(
                    filtered_vehicles,
                    x='Cost_per_km_EUR',
                    nbins=40,
                    title='Cost per km Distribution'
                )
                fig.add_vline(
                    x=filtered_vehicles['Cost_per_km_EUR'].median(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Median: €{filtered_vehicles['Cost_per_km_EUR'].median():.3f}"
                )
                fig.update_layout(
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(title='€/km', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(title='Count', showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top/Bottom Performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top 10 Efficient Vehicles")
            top_vehicles = filtered_vehicles.nlargest(10, 'Efficiency_km_per_L')[
                ['Vehicle', 'Fleet', 'Category', 'Efficiency_km_per_L', 'Total_Distance_km']
            ]
            st.dataframe(
                top_vehicles.style.format({
                    'Efficiency_km_per_L': '{:.2f}',
                    'Total_Distance_km': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("### ⚠️ Bottom 10 (Need Review)")
            bottom_vehicles = filtered_vehicles.nsmallest(10, 'Efficiency_km_per_L')[
                ['Vehicle', 'Fleet', 'Category', 'Efficiency_km_per_L', 'Total_Distance_km']
            ]
            st.dataframe(
                bottom_vehicles.style.format({
                    'Efficiency_km_per_L': '{:.2f}',
                    'Total_Distance_km': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # Complete Vehicle Table
        st.markdown("### 📋 Complete Vehicle Details")
        
        # Select columns to display
        display_cols = ['Vehicle', 'Fleet', 'Category', 'Fuel_Type', 'Brand_Model', 
                       'Total_Distance_km', 'Total_Fuel_L', 'Efficiency_km_per_L',
                       'Total_Cost_EUR', 'Cost_per_km_EUR', 'Total_CO2_kg']
        
        available_cols = [col for col in display_cols if col in filtered_vehicles.columns]
        vehicle_table = filtered_vehicles[available_cols].sort_values('Efficiency_km_per_L', ascending=False)
        
        st.dataframe(
            vehicle_table.style.format({
                'Total_Distance_km': '{:,.0f}',
                'Total_Fuel_L': '{:,.0f}',
                'Efficiency_km_per_L': '{:.2f}',
                'Total_Cost_EUR': '€{:,.2f}',
                'Cost_per_km_EUR': '€{:.3f}',
                'Total_CO2_kg': '{:,.1f}'
            }),
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Download button
        csv = vehicle_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"vehicle_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # PAGE: ANOMALY DASHBOARD
    # ========================================================================
    
    elif page == "⚠️ Anomaly Dashboard":
        st.markdown("# ⚠️ Anomaly Detection & Monitoring")
        st.markdown("Category-aware anomaly detection with intelligent thresholds")
        
        # Anomaly Overview
        total_records = len(data['daily'])
        anomaly_records = len(data['anomalies'])
        anomaly_rate = (anomaly_records / total_records * 100) if total_records > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            st.metric("Anomalies Detected", f"{anomaly_records:,}")
        with col3:
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        with col4:
            unique_vehicles = data['anomalies']['Vehicle'].nunique() if 'Vehicle' in data['anomalies'].columns else 0
            st.metric("Affected Vehicles", f"{unique_vehicles:,}")
        
        st.markdown('<div class="info-box"><strong>ℹ️ Note:</strong> High Idle detections are informational only. Focus on Low Efficiency, Unrealistic Efficiency, and Low Distance anomalies for operational improvements.</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Anomaly Type Breakdown
        st.markdown("### 📊 Anomaly Types Distribution")
        
        if 'Anomaly_Type' in data['anomalies'].columns:
            anomaly_counts = data['anomalies']['Anomaly_Type'].value_counts().reset_index()
            anomaly_counts.columns = ['Anomaly_Type', 'Count']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    anomaly_counts.sort_values('Count', ascending=False),
                    x='Anomaly_Type',
                    y='Count',
                    color='Anomaly_Type',
                    title='Anomaly Count by Type'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    anomaly_counts,
                    values='Count',
                    names='Anomaly_Type',
                    title='Anomaly Distribution'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=10)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Temporal Analysis
        st.markdown("### 📅 Anomaly Trends Over Time")
        
        if 'Date' in data['anomalies'].columns:
            daily_anomalies = data['anomalies'].groupby('Date').size().reset_index(name='Anomaly_Count')
            
            fig = create_trend_chart(daily_anomalies, 'Date', 'Anomaly_Count', 
                                    'Daily Anomaly Count', '#d62728')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Category Analysis
        st.markdown("### 📂 Anomalies by Category")
        
        if 'Category' in data['anomalies'].columns and 'Anomaly_Type' in data['anomalies'].columns:
            category_anomalies = data['anomalies'].groupby(['Category', 'Anomaly_Type']).size().reset_index(name='Count')
            
            fig = px.bar(
                category_anomalies,
                x='Category',
                y='Count',
                color='Anomaly_Type',
                title='Anomaly Distribution by Category',
                barmode='stack'
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Vehicle-Level Anomalies
        st.markdown("### 🚗 Vehicles with Most Anomalies")
        
        if 'Vehicle' in data['anomalies'].columns:
            vehicle_anomaly_counts = data['anomalies']['Vehicle'].value_counts().head(20).reset_index()
            vehicle_anomaly_counts.columns = ['Vehicle', 'Anomaly_Count']
            
            # Add category and fleet info if available
            if 'Category' in data['vehicle'].columns and 'Fleet' in data['vehicle'].columns:
                vehicle_info = data['vehicle'][['Vehicle', 'Category', 'Fleet']].drop_duplicates()
                vehicle_anomaly_counts = vehicle_anomaly_counts.merge(vehicle_info, on='Vehicle', how='left')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    vehicle_anomaly_counts.head(15),
                    y='Vehicle',
                    x='Anomaly_Count',
                    orientation='h',
                    color='Anomaly_Count',
                    color_continuous_scale='Reds',
                    title='Top 15 Vehicles by Anomaly Count'
                )
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(title='Anomaly Count', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(title='', showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Details")
                display_cols = ['Vehicle', 'Anomaly_Count']
                if 'Category' in vehicle_anomaly_counts.columns:
                    display_cols.append('Category')
                if 'Fleet' in vehicle_anomaly_counts.columns:
                    display_cols.append('Fleet')
                
                st.dataframe(
                    vehicle_anomaly_counts[display_cols].head(20),
                    use_container_width=True,
                    hide_index=True,
                    height=450
                )
        
        st.markdown("---")
        
        # Detailed Anomaly Table
        st.markdown("### 📋 Recent Anomalies (Last 100)")
        
        display_cols = ['Date', 'Vehicle', 'Fleet', 'Category', 'Anomaly_Type', 
                       'Efficiency_km_per_L', 'Distance_km', 'Idle_Ratio']
        available_cols = [col for col in display_cols if col in data['anomalies'].columns]
        
        recent_anomalies = data['anomalies'].sort_values('Date', ascending=False).head(100)[available_cols]
        
        st.dataframe(
            recent_anomalies.style.format({
                'Efficiency_km_per_L': '{:.2f}',
                'Distance_km': '{:.1f}',
                'Idle_Ratio': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Download anomalies
        csv = data['anomalies'].to_csv(index=False)
        st.download_button(
            label="📥 Download All Anomalies (CSV)",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # PAGE: DATA QUALITY
    # ========================================================================
    
    elif page == "🔍 Data Quality":
        st.markdown("# 🔍 Data Quality Diagnostics")
        st.markdown("Transparency into data processing and quality checks")
        
        # Quality Metrics
        total_input = len(data['daily']) + len(data['excluded'])
        excluded_count = len(data['excluded'])
        included_count = len(data['daily'])
        quality_rate = (included_count / total_input * 100) if total_input > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Input Records", f"{total_input:,}")
        with col2:
            st.metric("Records Analyzed", f"{included_count:,}")
        with col3:
            st.metric("Records Excluded", f"{excluded_count:,}")
        with col4:
            st.metric("Data Quality Rate", f"{quality_rate:.1f}%")
        
        if quality_rate >= 95:
            st.success("✅ Excellent data quality — over 95% of records passed validation")
        elif quality_rate >= 80:
            st.warning("⚠️ Good data quality — some records excluded due to validation rules")
        else:
            st.error("❌ Data quality concerns — significant portion of records excluded")
        
        st.markdown("---")
        
        # Exclusion Reasons
        st.markdown("### 📊 Exclusion Breakdown")
        
        if 'Exclusion_Reason' in data['excluded'].columns:
            exclusion_counts = data['excluded']['Exclusion_Reason'].value_counts().reset_index()
            exclusion_counts.columns = ['Reason', 'Count']
            exclusion_counts['Percentage'] = (exclusion_counts['Count'] / excluded_count * 100)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    exclusion_counts.sort_values('Count', ascending=False),
                    x='Reason',
                    y='Count',
                    color='Count',
                    color_continuous_scale='Oranges',
                    title='Records Excluded by Reason'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Summary")
                st.dataframe(
                    exclusion_counts.style.format({'Percentage': '{:.2f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
        
        st.markdown("---")
        
        # Category Coverage
        st.markdown("### 📂 Data Coverage by Category")
        
        if 'Category' in data['vehicle'].columns:
            category_coverage = data['vehicle']['Category'].value_counts().reset_index()
            category_coverage.columns = ['Category', 'Vehicle_Count']
            category_coverage['Percentage'] = (category_coverage['Vehicle_Count'] / total_vehicles * 100)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.pie(
                    category_coverage,
                    values='Vehicle_Count',
                    names='Category',
                    title='Vehicle Distribution by Category',
                    color='Category',
                    color_discrete_map={cat: get_category_color(cat) for cat in category_coverage['Category'].unique()}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Coverage")
                st.dataframe(
                    category_coverage.style.format({'Percentage': '{:.1f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
        
        st.markdown("---")
        
        # Fuel Type Coverage
        st.markdown("### ⛽ Fuel Type Distribution")
        
        if 'Fuel_Type' in data['vehicle'].columns:
            fuel_coverage = data['vehicle']['Fuel_Type'].value_counts().reset_index()
            fuel_coverage.columns = ['Fuel_Type', 'Vehicle_Count']
            
            fig = px.bar(
                fuel_coverage.sort_values('Vehicle_Count', ascending=False),
                x='Fuel_Type',
                y='Vehicle_Count',
                color='Fuel_Type',
                title='Vehicles by Fuel Type'
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Data Completeness
        st.markdown("### ✅ Data Completeness Check")
        
        # Calculate completeness for key metrics
        completeness_metrics = {}
        key_columns = ['Efficiency_km_per_L', 'Total_Distance_km', 'Total_Fuel_L', 
                      'Total_Cost_EUR', 'Total_CO2_kg']
        
        for col in key_columns:
            if col in data['vehicle'].columns:
                non_null = data['vehicle'][col].notna().sum()
                completeness = (non_null / len(data['vehicle']) * 100)
                completeness_metrics[col] = completeness
        
        if completeness_metrics:
            completeness_df = pd.DataFrame(list(completeness_metrics.items()), 
                                          columns=['Metric', 'Completeness_%'])
            completeness_df = completeness_df.sort_values('Completeness_%', ascending=False)
            
            fig = px.bar(
                completeness_df,
                x='Metric',
                y='Completeness_%',
                color='Completeness_%',
                color_continuous_scale='Greens',
                title='Data Completeness by Metric',
                range_color=[0, 100]
            )
            fig.add_hline(y=95, line_dash="dash", line_color="red", 
                         annotation_text="95% threshold")
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 105])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Excluded Records Sample
        st.markdown("### 📋 Sample of Excluded Records")
        
        if len(data['excluded']) > 0:
            sample_size = min(100, len(data['excluded']))
            st.markdown(f"Showing {sample_size} most recent excluded records")
            
            display_cols = [col for col in data['excluded'].columns if col in ['Date', 'Vehicle', 'Fleet', 
                           'Distance_km', 'Fuel_L', 'Efficiency_km_per_L', 'Exclusion_Reason']]
            
            excluded_sample = data['excluded'][display_cols].head(sample_size)
            st.dataframe(excluded_sample, use_container_width=True, hide_index=True, height=400)
            
            # Download excluded records
            csv = data['excluded'].to_csv(index=False)
            st.download_button(
                label="📥 Download All Excluded Records (CSV)",
                data=csv,
                file_name=f"excluded_records_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No records were excluded — perfect data quality!")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
    
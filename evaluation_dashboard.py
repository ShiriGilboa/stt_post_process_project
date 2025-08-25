import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import difflib
from collections import defaultdict
import os
import hashlib
import sys
import re

# Add the dev directory to Python path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'stt_post_process_improvements', 'dev'))

try:
    from utils import normalize_text
    NORMALIZATION_AVAILABLE = True
except ImportError as e:
    st.warning(f"Could not import normalization functions: {e}. Text will be displayed as-is.")
    NORMALIZATION_AVAILABLE = False
    # Fallback normalize function
    def normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()

        def replace_number(match):
            try:
                return num2words.num2words(int(match.group()))
            except Exception:
                return match.group()

        # Replace numbers with words **before** removing special characters
        text = re.sub(r'\d+', replace_number, text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# Authentication configuration
# Priority: 1) Streamlit secrets, 2) Environment variable, 3) Fallback
try:
    DASHBOARD_PASSWORD = st.secrets["STREAMLIT_PASSWORD"]
except (KeyError, FileNotFoundError):
    DASHBOARD_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "stt_dashboard_2024")  # Use env var or fallback

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256(DASHBOARD_PASSWORD.encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Create centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown('<h2 class="login-header">🔐 Dashboard Access</h2>', unsafe_allow_html=True)
        st.markdown("Please enter the password to access the STT Enhancement Evaluation Dashboard:")
        
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            help="Contact the administrator if you don't have the password"
        )
        
        if "password_correct" in st.session_state:
            if not st.session_state["password_correct"]:
                st.error("😞 Password incorrect. Please try again.")
        
        st.markdown('<p class="login-info">This dashboard contains sensitive evaluation data and requires authentication.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

# Page configuration
st.set_page_config(
    page_title="STT Enhancement Evaluation Dashboard",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication before showing the dashboard
if not check_password():
    st.stop()  # Do not continue if password is not correct

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    /* Login form styling */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
    }
    
    .login-header {
        text-align: center;
        color: #333;
        margin-bottom: 1.5rem;
    }
    
    .login-info {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-top: 1rem;
    }
    
    /* Pipeline name styling - smaller fonts */
    .small-pipeline-name {
        font-size: 0.85rem !important;
        line-height: 1.2 !important;
    }
    
    /* Streamlit metric labels with smaller font */
    [data-testid="metric-container"] label {
        font-size: 0.65rem !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 0.7rem !important;
    }
    
    /* Executive Summary metrics - even smaller fonts */
    [data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 0.65rem !important;
        font-weight: 600 !important;
    }
    
    /* Metric delta text smaller */
    [data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 0.6rem !important;
    }
    
    /* Make metric labels even smaller */
    [data-testid="metric-container"] > label {
        font-size: 0.6rem !important;
        font-weight: 500 !important;
    }
    
    /* Target all text in metric containers */
    [data-testid="metric-container"] * {
        font-size: 0.65rem !important;
    }
    
    /* Override any large text in metrics */
    [data-testid="metric-container"] p,
    [data-testid="metric-container"] span,
    [data-testid="metric-container"] div {
        font-size: 0.65rem !important;
        line-height: 1.1 !important;
    }
    
    /* Expander headers with smaller font for pipeline names */
    .streamlit-expander .streamlit-expanderHeader p {
        font-size: 0.9rem !important;
        line-height: 1.3 !important;
    }
    
    /* Selectbox options with smaller font */
    .stSelectbox label {
        font-size: 0.9rem !important;
    }
    
    /* Chart legend text smaller */
    .plotly .legend text {
        font-size: 11px !important;
    }
    
    /* Text highlighting for sample analysis */
    .transcript-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.6;
    }
    
    .ground-truth-box {
        border-left: 4px solid #007bff;
    }
    
    .baseline-box {
        border-left: 4px solid #ffc107;
    }
    
    .pipeline-box {
        border-left: 4px solid #17a2b8;
    }
    
    .diff-insert {
        background-color: #d4edda;
        color: #155724;
        padding: 3px 6px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .diff-delete {
        background-color: #f8d7da;
        color: #721c24;
        padding: 3px 6px;
        border-radius: 4px;
        text-decoration: line-through;
        font-weight: 500;
    }
    
    .pipeline-result {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .pipeline-name {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 10px;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
    }
    
    .wer-improvement {
        color: #28a745;
        font-weight: bold;
    }
    
    .wer-degradation {
        color: #dc3545;
        font-weight: bold;
    }
    
    .wer-same {
        color: #6c757d;
        font-weight: bold;
    }
    
    .prompt-box {
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the evaluation data"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'stt_post_process_improvements','dev', 'after_evaluation', 'evaluation_results_unified.csv')
        df = pd.read_csv(csv_path)        
        return df
    except FileNotFoundError as e:
        st.error(f"evaluation_results_unified.csv not found. Please ensure the file exists in the same directory. {e}")
        return None

def get_friendly_pipeline_name(technical_name):
    """Convert technical pipeline names to user-friendly names"""
    name_mapping = {
        'FixTranscriptByLLMPipeline': 'Direct LLM Correction',
        'GenerateWhisperPromptPipeline': 'Smart Context Enhancement', 
        'GenerateNamesPipeline': 'Name Recognition & Correction',
        'GenerateTopicPipeline': 'Topic Identification'
    }
    return name_mapping.get(technical_name, technical_name)

def get_pipeline_description(technical_name):
    """Get detailed description of what each pipeline does"""
    descriptions = {
        'FixTranscriptByLLMPipeline': 'Directly corrects STT errors (spelling, casing, punctuation) using LLM without additional context',
        'GenerateWhisperPromptPipeline': 'Advanced multi-agent system that extracts topics, names, and jargon to generate contextual prompts for Whisper',
        'GenerateNamesPipeline': 'Identifies and corrects person names using topic-aware fuzzy matching against domain-specific databases',
        'GenerateTopicPipeline': 'Extracts the main topic/domain (2-5 words) to provide context for other enhancement methods'
    }
    return descriptions.get(technical_name, 'No description available')

def calculate_pipeline_stats(df):
    """Calculate comprehensive statistics for each pipeline"""
    pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                'GenerateNamesPipeline', 'GenerateTopicPipeline']
    
    stats = []
    
    for pipeline in pipelines:
        wer_col = f'{pipeline}_wer'
        if wer_col in df.columns:
            # Remove NaN values for comparison
            valid_mask = (~df['stt_raw_wer'].isna()) & (~df[wer_col].isna())
            valid_df = df[valid_mask]
            
            if len(valid_df) > 0:
                # Calculate improvement categories
                improved = (valid_df[wer_col] < valid_df['stt_raw_wer']).sum()
                same = (np.abs(valid_df[wer_col] - valid_df['stt_raw_wer']) < 1e-6).sum()
                degraded = (valid_df[wer_col] > valid_df['stt_raw_wer']).sum()
                total = len(valid_df)
                
                # Calculate WER statistics
                baseline_wer = valid_df['stt_raw_wer'].mean()
                pipeline_wer = valid_df[wer_col].mean()
                wer_improvement = baseline_wer - pipeline_wer
                relative_improvement = (wer_improvement / baseline_wer) * 100 if baseline_wer > 0 else 0
                
                stats.append({
                    'Pipeline': get_friendly_pipeline_name(pipeline),
                    'Technical_Name': pipeline,
                    'Description': get_pipeline_description(pipeline),
                    'Total_Segments': total,
                    'Improved': improved,
                    'Same': same,
                    'Degraded': degraded,
                    'Improved_%': (improved / total) * 100,
                    'Same_%': (same / total) * 100,
                    'Degraded_%': (degraded / total) * 100,
                    'Baseline_WER': baseline_wer,
                    'Pipeline_WER': pipeline_wer,
                    'WER_Improvement': wer_improvement,
                    'Relative_Improvement_%': relative_improvement
                })
    
    return pd.DataFrame(stats)

def create_pipeline_comparison_chart(stats_df):
    """Create comprehensive pipeline comparison visualizations"""
    
    # Create individual charts instead of subplots for better control
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Improvement Rate Comparison
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                x=stats_df['Pipeline'],
                y=stats_df['Improved_%'],
                name='Improvement Rate',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(stats_df)],
                text=[f"{val:.1f}%" for val in stats_df['Improved_%']],
                textposition='outside',
                textfont=dict(size=12, family='Arial')
            )
        )
        fig1.update_layout(
            title="Improvement Rate Comparison",
            xaxis_title="Pipeline",
            yaxis_title="Improvement Rate (%)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 3. Segment Distribution (Stacked Bar)
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                x=stats_df['Pipeline'],
                y=stats_df['Improved_%'],
                name='Improved',
                marker_color='#2ca02c',
                text=[f"{val:.1f}%" for val in stats_df['Improved_%']],
                textposition='inside',
                textfont=dict(color='white', size=10, family='Arial')
            )
        )
        fig3.add_trace(
            go.Bar(
                x=stats_df['Pipeline'],
                y=stats_df['Same_%'],
                name='Same',
                marker_color='#7f7f7f',
                text=[f"{val:.1f}%" for val in stats_df['Same_%']],
                textposition='inside',
                textfont=dict(color='white', size=10, family='Arial')
            )
        )
        fig3.add_trace(
            go.Bar(
                x=stats_df['Pipeline'],
                y=stats_df['Degraded_%'],
                name='Degraded',
                marker_color='#d62728',
                text=[f"{val:.1f}%" for val in stats_df['Degraded_%']],
                textposition='inside',
                textfont=dict(color='white', size=10, family='Arial')
            )
        )
        fig3.update_layout(
            title="Segment Distribution",
            xaxis_title="Pipeline",
            yaxis_title="Percentage",
            barmode='stack',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # 2. WER Performance
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=stats_df['Pipeline'],
                y=stats_df['Baseline_WER'],
                name='Baseline WER',
                marker_color='#d62728',
                opacity=0.7
            )
        )
        fig2.add_trace(
            go.Bar(
                x=stats_df['Pipeline'],
                y=stats_df['Pipeline_WER'],
                name='Pipeline WER',
                marker_color='#1f77b4'
            )
        )
        fig2.update_layout(
            title="WER Performance",
            xaxis_title="Pipeline",
            yaxis_title="Word Error Rate",
            barmode='group',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # 4. Risk vs Reward Analysis
        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(
                x=stats_df['Degraded_%'],
                y=stats_df['Improved_%'],
                mode='markers+text',
                text=stats_df['Pipeline'],
                textposition="top center",
                marker=dict(
                    size=stats_df['Total_Segments'] / 10,
                    color=stats_df['Relative_Improvement_%'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Relative Improvement %")
                ),
                name='Pipelines'
            )
        )
        fig4.update_layout(
            title="Risk vs Reward Analysis",
            xaxis_title="Degradation Risk (%)",
            yaxis_title="Improvement Rate (%)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig4, use_container_width=True)

def highlight_differences(text1, text2):
    """Create HTML highlighting differences between two texts"""
    if pd.isna(text1) or pd.isna(text2):
        return text1, text2
    
    # Split into words for better comparison
    words1 = str(text1).split()
    words2 = str(text2).split()
    
    # Use SequenceMatcher for word-level comparison
    matcher = difflib.SequenceMatcher(None, words1, words2)
    
    def format_text(words, opcodes, is_first=True):
        result = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                result.extend(words[i1:i2] if is_first else words[j1:j2])
            elif tag == 'delete' and is_first:
                for word in words[i1:i2]:
                    result.append(f'<span class="diff-delete">{word}</span>')
            elif tag == 'insert' and not is_first:
                for word in words[j1:j2]:
                    result.append(f'<span class="diff-insert">{word}</span>')
            elif tag == 'replace':
                if is_first:
                    for word in words[i1:i2]:
                        result.append(f'<span class="diff-delete">{word}</span>')
                else:
                    for word in words[j1:j2]:
                        result.append(f'<span class="diff-insert">{word}</span>')
        return ' '.join(result)
    
    opcodes = matcher.get_opcodes()
    formatted1 = format_text(words1, opcodes, True)
    formatted2 = format_text(words2, opcodes, False)
    
    return formatted1, formatted2

def get_short_pipeline_name(technical_name):
    """Convert technical pipeline names to very short names for metrics display"""
    short_mapping = {
        'FixTranscriptByLLMPipeline': 'Direct LLM',
        'GenerateWhisperPromptPipeline': 'Smart Context', 
        'GenerateNamesPipeline': 'Name Recognition',
        'GenerateTopicPipeline': 'Topic ID'
    }
    return short_mapping.get(technical_name, technical_name)

def main():
    # Add logout button to sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", help="Click to logout and return to login screen"):
            # Clear the authentication state
            for key in list(st.session_state.keys()):
                if key.startswith("password"):
                    del st.session_state[key]
            st.rerun()
    
    # Header
    st.markdown('<h1 class="main-header">🎤 STT Enhancement Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Agent LLM Pipeline Performance Analysis for Speech-to-Text Enhancement**")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis View",
        ["📊 Main Dashboard", "🔍 Pipeline Comparison", "📝 Sample Analysis", "🎯 Detailed Metrics"]
    )
    
    if page == "📊 Main Dashboard":
        show_main_dashboard(df)
    elif page == "🔍 Pipeline Comparison":
        show_pipeline_comparison(df)
    elif page == "📝 Sample Analysis":
        show_sample_analysis(df)
    elif page == "🎯 Detailed Metrics":
        show_detailed_metrics(df)

def show_main_dashboard(df):
    st.header("📊 Executive Summary")
    
    # Calculate overall statistics
    stats_df = calculate_pipeline_stats(df)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Segments", 
            value=f"{len(df)}",
            help=f"Across {df['video_id_extracted'].nunique()} NBA videos"
        )
    
    with col2:
        best_pipeline = stats_df.loc[stats_df['Improved_%'].idxmax()]
        # Use shorter name for display
        short_name = get_short_pipeline_name(best_pipeline['Technical_Name'])
        st.metric(
            label="Best Pipeline",
            value=short_name,
            delta=f"{best_pipeline['Improved_%']:.1f}% improvement rate",
            delta_color="normal",
            help=best_pipeline['Description']
        )
    
    with col3:
        safest_pipeline = stats_df.loc[stats_df['Degraded_%'].idxmin()]
        short_safe_name = get_short_pipeline_name(safest_pipeline['Technical_Name'])
        st.metric(
            label="Safest Pipeline",
            value=short_safe_name,
            delta=f"{safest_pipeline['Degraded_%']:.1f}% degradation risk",
            delta_color="inverse",
            help=safest_pipeline['Description']
        )
    
    with col4:
        avg_baseline_wer = df['stt_raw_wer'].mean()
        st.metric(
            label="Baseline WER",
            value=f"{avg_baseline_wer:.3f}",
            help="Average Word Error Rate across all segments"
        )
    
    st.markdown("---")
    
    # Pipeline descriptions section
    st.subheader("🔧 Pipeline Descriptions")
    
    for _, row in stats_df.iterrows():
        with st.expander(f"**{row['Pipeline']}** - {row['Improved_%']:.1f}% improvement rate", expanded=False):
            st.write(row['Description'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Segments", row['Total_Segments'])
            with col2:
                st.metric("Average WER", f"{row['Pipeline_WER']:.4f}")
            with col3:
                improvement = row['WER_Improvement']
                if improvement > 0:
                    st.metric("WER Reduction", f"{improvement:.4f}", delta="Improvement", delta_color="normal")
                elif improvement < 0:
                    st.metric("WER Change", f"{improvement:.4f}", delta="Degradation", delta_color="inverse")
                else:
                    st.metric("WER Change", "0.0000", delta="No change", delta_color="off")
    
    st.markdown("---")
    
    # Main comparison chart
    st.subheader("Pipeline Performance Overview")
    create_pipeline_comparison_chart(stats_df)
    
    # Performance summary table
    st.subheader("Performance Summary Table")
    display_df = stats_df.copy()
    display_df['Improved_%'] = display_df['Improved_%'].round(1)
    display_df['Degraded_%'] = display_df['Degraded_%'].round(1)
    display_df['Baseline_WER'] = display_df['Baseline_WER'].round(4)
    display_df['Pipeline_WER'] = display_df['Pipeline_WER'].round(4)
    display_df['Relative_Improvement_%'] = display_df['Relative_Improvement_%'].round(2)
    
    st.dataframe(
        display_df[['Pipeline', 'Total_Segments', 'Improved_%', 'Degraded_%', 
                   'Baseline_WER', 'Pipeline_WER', 'Relative_Improvement_%']],
        use_container_width=True
    )

def show_pipeline_comparison(df):
    st.header("🔍 Detailed Pipeline Comparison")
    
    # Pipeline selection with friendly names
    technical_pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                          'GenerateNamesPipeline', 'GenerateTopicPipeline']
    
    # Create options with friendly names but keep technical names as values
    pipeline_options = {get_friendly_pipeline_name(p): p for p in technical_pipelines}
    
    selected_friendly = st.multiselect(
        "Select Pipelines to Compare",
        list(pipeline_options.keys()),
        default=list(pipeline_options.keys())[:2],
        help="Choose which enhancement methods to compare"
    )
    
    # Convert back to technical names for processing
    selected_pipelines = [pipeline_options[name] for name in selected_friendly]
    
    if len(selected_pipelines) < 2:
        st.warning("Please select at least 2 pipelines for comparison.")
        return
    
    # Show selected pipeline descriptions
    st.subheader("Selected Pipeline Descriptions")
    for friendly_name in selected_friendly:
        technical_name = pipeline_options[friendly_name]
        with st.expander(f"**{friendly_name}**", expanded=False):
            st.write(get_pipeline_description(technical_name))
    
    st.markdown("---")
    
    # WER distribution comparison
    st.subheader("WER Distribution Comparison")
    
    fig = go.Figure()
    
    for pipeline in selected_pipelines:
        wer_col = f'{pipeline}_wer'
        if wer_col in df.columns:
            valid_data = df[df[wer_col].notna()][wer_col]
            friendly_name = get_friendly_pipeline_name(pipeline)
            fig.add_trace(go.Histogram(
                x=valid_data,
                name=friendly_name,
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        title="WER Distribution by Pipeline",
        xaxis_title="Word Error Rate",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Head-to-head comparison
    if len(selected_pipelines) == 2:
        st.subheader("Head-to-Head Comparison")
        
        pipeline1, pipeline2 = selected_pipelines
        friendly1 = get_friendly_pipeline_name(pipeline1)
        friendly2 = get_friendly_pipeline_name(pipeline2)
        
        wer1_col = f'{pipeline1}_wer'
        wer2_col = f'{pipeline2}_wer'
        
        # Filter valid comparisons
        comparison_df = df[(df[wer1_col].notna()) & (df[wer2_col].notna())].copy()
        
        # Calculate win/loss/tie
        pipeline1_wins = (comparison_df[wer1_col] < comparison_df[wer2_col]).sum()
        pipeline2_wins = (comparison_df[wer2_col] < comparison_df[wer1_col]).sum()
        ties = len(comparison_df) - pipeline1_wins - pipeline2_wins
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{friendly1} Wins", pipeline1_wins)
        
        with col2:
            st.metric("Ties", ties)
        
        with col3:
            st.metric(f"{friendly2} Wins", pipeline2_wins)
        
        # Scatter plot comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison_df[wer1_col],
            y=comparison_df[wer2_col],
            mode='markers',
            marker=dict(
                color=comparison_df['stt_raw_wer'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Baseline WER")
            ),
            text=comparison_df['video_id_extracted'],
            hovertemplate=f"{friendly1}: %{{x:.3f}}<br>{friendly2}: %{{y:.3f}}<br>Video: %{{text}}<extra></extra>"
        ))
        
        # Add diagonal line
        min_wer = min(comparison_df[wer1_col].min(), comparison_df[wer2_col].min())
        max_wer = max(comparison_df[wer1_col].max(), comparison_df[wer2_col].max())
        fig.add_trace(go.Scatter(
            x=[min_wer, max_wer],
            y=[min_wer, max_wer],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Equal Performance'
        ))
        
        fig.update_layout(
            title=f"{friendly1} vs {friendly2} Performance",
            xaxis_title=f"{friendly1} WER",
            yaxis_title=f"{friendly2} WER"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_sample_analysis(df):
    st.header("📝 Sample-by-Sample Analysis")
    
    # Sample selection
    video_options = ['All Videos'] + sorted(df['video_id_extracted'].unique())
    selected_video = st.selectbox("Select Video", video_options)
    
    if selected_video != 'All Videos':
        filtered_df = df[df['video_id_extracted'] == selected_video]
    else:
        filtered_df = df
    
    # Sample selection
    sample_idx = st.selectbox(
        "Select Sample",
        range(len(filtered_df)),
        format_func=lambda x: f"Sample {x+1}: {filtered_df.iloc[x]['segment_filename']}"
    )
    
    if sample_idx is not None:
        sample = filtered_df.iloc[sample_idx]
        
        # Sample information
        st.subheader("📋 Sample Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Segment", sample['segment_filename'])
            st.metric("Video ID", sample['video_id_extracted'])
        
        with col2:
            st.metric("Start Time", f"{sample['start']:.1f}s")
            st.metric("End Time", f"{sample['end']:.1f}s")
        
        with col3:
            st.metric("Duration", f"{sample['end'] - sample['start']:.1f}s")
            st.metric("Baseline WER", f"{sample['stt_raw_wer']:.4f}")
        
        st.markdown("---")
        
        # Ground truth
        st.subheader("🎯 Ground Truth")
        
        if 'transcript' in sample:
            original_transcript = sample['transcript']
            normalized_gt = normalize_text(original_transcript)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original:**")
                st.markdown(f'<div class="transcript-box ground-truth-box">{original_transcript}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown("**Normalized (used for WER):**")
                st.markdown(f'<div class="transcript-box ground-truth-box">{normalized_gt}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="transcript-box ground-truth-box"><em>No ground truth available</em></div>', unsafe_allow_html=True)
        
        # Baseline (raw STT)
        st.subheader("📄 Baseline (Raw Whisper)")
        
        # Show baseline WER
        if 'stt_raw_wer' in sample and pd.notna(sample['stt_raw_wer']):
            st.metric("Baseline WER", f"{sample['stt_raw_wer']:.4f}")
        
        baseline_text = ""
        if 'stt_raw_norm' in sample and pd.notna(sample['stt_raw_norm']):
            baseline_text = sample['stt_raw_norm']
            # Use normalized ground truth for comparison (same normalization as used for WER)
            if 'transcript' in sample:
                normalized_ground_truth = normalize_text(sample['transcript'])
                # Highlight differences from normalized ground truth
                gt_highlighted, baseline_highlighted = highlight_differences(normalized_ground_truth, baseline_text)
                st.markdown(f'<div class="transcript-box baseline-box">{baseline_highlighted}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="transcript-box baseline-box">{baseline_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="transcript-box baseline-box"><em>Baseline transcription not available</em></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # All Pipeline Results
        st.subheader("🔄 All Pipeline Results")
        
        pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                    'GenerateNamesPipeline', 'GenerateTopicPipeline']
        
        for pipeline in pipelines:
            friendly_name = get_friendly_pipeline_name(pipeline)
            description = get_pipeline_description(pipeline)
            
            with st.expander(f"🔧 {friendly_name}", expanded=True):
                # Show pipeline description
                st.markdown(f"*{description}*")
                st.markdown("---")
                
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    # Performance metrics
                    pipeline_wer_col = f'{pipeline}_wer'
                    if pipeline_wer_col in sample and pd.notna(sample[pipeline_wer_col]):
                        pipeline_wer = sample[pipeline_wer_col]
                        improvement = sample['stt_raw_wer'] - pipeline_wer
                        
                        st.metric("WER", f"{pipeline_wer:.4f}")
                        
                        if improvement > 0.001:
                            st.markdown(f'<div class="wer-improvement">✅ Improved by {improvement:.4f}</div>', unsafe_allow_html=True)
                        elif improvement < -0.001:
                            st.markdown(f'<div class="wer-degradation">❌ Degraded by {abs(improvement):.4f}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="wer-same">➖ No change</div>', unsafe_allow_html=True)
                    else:
                        st.write("*No WER data available*")
                
                with col1:
                    # Pipeline result text (already normalized in CSV)
                    pipeline_norm_col = f'{pipeline}_norm'
                    if pipeline_norm_col in sample and pd.notna(sample[pipeline_norm_col]):
                        pipeline_text = sample[pipeline_norm_col]
                        
                        # Highlight differences from baseline (both are already normalized)
                        if baseline_text:
                            baseline_highlighted2, pipeline_highlighted = highlight_differences(baseline_text, pipeline_text)
                            st.markdown(f'<div class="transcript-box pipeline-box">{pipeline_highlighted}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="transcript-box pipeline-box">{pipeline_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="transcript-box pipeline-box"><em>No transcription available</em></div>', unsafe_allow_html=True)
                
                # Show initial prompt if available (especially relevant for Smart Context Enhancement)
                pipeline_prompt_col = f'{pipeline}_initial_prompt'
                if pipeline_prompt_col in sample and pd.notna(sample[pipeline_prompt_col]):
                    st.markdown("**🎯 Generated Context/Prompt:**")
                    st.markdown(f'<div class="prompt-box">{sample[pipeline_prompt_col]}</div>', unsafe_allow_html=True)

def show_video_performance(df):
    st.header("📈 Video-by-Video Performance Analysis")
    
    # Calculate video-level statistics
    video_stats = []
    
    for video_id in df['video_id_extracted'].unique():
        video_df = df[df['video_id_extracted'] == video_id]
        
        video_stat = {'Video_ID': video_id, 'Total_Segments': len(video_df)}
        
        # Calculate stats for each pipeline
        pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                    'GenerateNamesPipeline', 'GenerateTopicPipeline']
        
        for pipeline in pipelines:
            wer_col = f'{pipeline}_wer'
            if wer_col in video_df.columns:
                valid_mask = (~video_df['stt_raw_wer'].isna()) & (~video_df[wer_col].isna())
                valid_df = video_df[valid_mask]
                
                if len(valid_df) > 0:
                    improved = (valid_df[wer_col] < valid_df['stt_raw_wer']).sum()
                    total = len(valid_df)
                    friendly_name = get_friendly_pipeline_name(pipeline)
                    video_stat[f'{friendly_name}_Improvement_%'] = (improved / total) * 100
                    video_stat[f'{friendly_name}_Baseline_WER'] = valid_df['stt_raw_wer'].mean()
                    video_stat[f'{friendly_name}_Pipeline_WER'] = valid_df[wer_col].mean()
        
        video_stats.append(video_stat)
    
    video_stats_df = pd.DataFrame(video_stats)
    
    # Video performance heatmap
    st.subheader("Improvement Rate by Video")
    
    improvement_cols = [col for col in video_stats_df.columns if col.endswith('_Improvement_%')]
    
    if improvement_cols:
        heatmap_data = video_stats_df.set_index('Video_ID')[improvement_cols]
        heatmap_data.columns = [col.replace('_Improvement_%', '') for col in heatmap_data.columns]
        
        fig = px.imshow(
            heatmap_data.T,
            title="Improvement Rate Heatmap (% of segments improved)",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis_title="Video ID",
            yaxis_title="Pipeline"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Video difficulty analysis
    st.subheader("Video Difficulty Analysis")
    
    fig = go.Figure()
    
    for pipeline in ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                    'GenerateNamesPipeline', 'GenerateTopicPipeline']:
        friendly_name = get_friendly_pipeline_name(pipeline)
        improvement_col = f'{friendly_name}_Improvement_%'
        baseline_col = f'{friendly_name}_Baseline_WER'
        
        if improvement_col in video_stats_df.columns and baseline_col in video_stats_df.columns:
            fig.add_trace(go.Scatter(
                x=video_stats_df[baseline_col],
                y=video_stats_df[improvement_col],
                mode='markers+text',
                text=video_stats_df['Video_ID'],
                textposition="top center",
                name=friendly_name,
                marker=dict(size=10)
            ))
    
    fig.update_layout(
        title="Video Difficulty vs Pipeline Performance",
        xaxis_title="Baseline WER (Video Difficulty)",
        yaxis_title="Improvement Rate (%)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed video table
    st.subheader("Detailed Video Statistics")
    st.dataframe(video_stats_df, use_container_width=True)



def show_detailed_metrics(df):
    st.header("🎯 Detailed Performance Metrics")
    
    pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                'GenerateNamesPipeline', 'GenerateTopicPipeline']
    
    # Baseline WER Distribution Analysis
    st.subheader("📊 Baseline WER Distribution Analysis")
    
    baseline_wer = df['stt_raw_wer'].dropna()
    zero_wer_count = (baseline_wer == 0).sum()
    perfect_segments_pct = (zero_wer_count / len(baseline_wer)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", len(baseline_wer))
        st.metric("Mean WER", f"{baseline_wer.mean():.4f}")
    with col2:
        st.metric("Perfect Segments (WER=0)", zero_wer_count)
        st.metric("Perfect %", f"{perfect_segments_pct:.1f}%")
    with col3:
        st.metric("Improvable Segments", len(baseline_wer) - zero_wer_count)
        st.metric("Improvable %", f"{100-perfect_segments_pct:.1f}%")
    
    # Baseline distribution histogram
    fig = px.histogram(
        x=baseline_wer,
        nbins=30,
        title="Baseline WER Distribution - Shows Perfect vs Improvable Segments",
        labels={'x': 'Word Error Rate', 'y': 'Count'},
        marginal="box"
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", 
                  annotation_text=f"Perfect (WER=0): {zero_wer_count} segments")
    fig.add_vline(x=baseline_wer.mean(), line_dash="dash", line_color="green", 
                  annotation_text=f"Mean: {baseline_wer.mean():.3f}")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    **Key Insight:** {perfect_segments_pct:.1f}% of segments have perfect baseline transcriptions (WER=0). 
    These cannot be improved by any pipeline. The analysis below focuses on the {100-perfect_segments_pct:.1f}% 
    of segments where improvement is actually possible.
    """)
    
    st.markdown("---")
    
    
    # Analysis excluding perfect transcriptions (WER = 0)
    st.subheader("🎯 Analysis: Improvable Segments Only (Baseline WER > 0)")
    
    non_zero_mask = df['stt_raw_wer'] > 0
    improvable_df = df[non_zero_mask]
    
    if len(improvable_df) > 0:
        st.markdown(f"""
        **Focus on Real Impact:** This analysis excludes {zero_wer_count} perfect segments and focuses on 
        the {len(improvable_df)} segments where Whisper baseline had errors and improvement is possible.
        """)
        
        # Recalculate statistics for improvable segments only
        improvable_stats = []
        
        for pipeline in pipelines:
            wer_col = f'{pipeline}_wer'
            if wer_col in improvable_df.columns:
                valid_mask = (~improvable_df['stt_raw_wer'].isna()) & (~improvable_df[wer_col].isna())
                valid_df = improvable_df[valid_mask]
                
                if len(valid_df) > 0:
                    improved = (valid_df[wer_col] < valid_df['stt_raw_wer']).sum()
                    same = (np.abs(valid_df[wer_col] - valid_df['stt_raw_wer']) < 1e-6).sum()
                    degraded = (valid_df[wer_col] > valid_df['stt_raw_wer']).sum()
                    total = len(valid_df)
                    
                    baseline_wer_mean = valid_df['stt_raw_wer'].mean()
                    pipeline_wer_mean = valid_df[wer_col].mean()
                    wer_reduction = baseline_wer_mean - pipeline_wer_mean
                    relative_reduction = (wer_reduction / baseline_wer_mean) * 100 if baseline_wer_mean > 0 else 0
                    
                    improvable_stats.append({
                        'Pipeline': get_friendly_pipeline_name(pipeline),
                        'Total_Segments': total,
                        'Improved': improved,
                        'Improved_%': (improved / total) * 100,
                        'Same': same,
                        'Same_%': (same / total) * 100,
                        'Degraded': degraded,
                        'Degraded_%': (degraded / total) * 100,
                        'Baseline_WER': baseline_wer_mean,
                        'Pipeline_WER': pipeline_wer_mean,
                        'WER_Reduction': wer_reduction,
                        'Relative_Reduction_%': relative_reduction
                    })
        
        if improvable_stats:
            improvable_stats_df = pd.DataFrame(improvable_stats)
            
            # Comparison: All segments vs Improvable segments
            st.subheader("📊 Impact Comparison: All vs Improvable Segments")
            
            all_stats_df = calculate_pipeline_stats(df)
            comparison_data = []
            
            for _, row in improvable_stats_df.iterrows():
                pipeline_name = row['Pipeline']
                all_row = all_stats_df[all_stats_df['Pipeline'] == pipeline_name].iloc[0]
                
                comparison_data.extend([
                    {'Pipeline': pipeline_name, 'Dataset': 'All Segments (WER=0 && WER>0)', 'Improvement_%': all_row['Improved_%']},
                    {'Pipeline': pipeline_name, 'Dataset': 'Improvable Only (WER>0)', 'Improvement_%': row['Improved_%']}
                ])
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='Pipeline',
                y='Improvement_%',
                color='Dataset',
                barmode='group',
                title="Improvement Rate: Including vs Excluding Perfect Segments",
                labels={'Improvement_%': 'Improvement Rate (%)'},
                color_discrete_map={'All Segments': '#ff7f0e', 'Improvable Only': '#2ca02c'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # WER distributions for improvable segments only
            st.subheader("📈 WER Distributions: Improvable Segments Only")
            
            fig = go.Figure()
            
            # Baseline for improvable segments
            improvable_baseline = improvable_df['stt_raw_wer'].dropna()
            fig.add_trace(go.Histogram(
                x=improvable_baseline,
                name="Baseline (Improvable)",
                opacity=0.7,
                nbinsx=20,
                marker_color='red'
            ))
            
            # Pipeline distributions for improvable segments
            for i, pipeline in enumerate(pipelines):
                wer_col = f'{pipeline}_wer'
                if wer_col in improvable_df.columns:
                    pipeline_wer = improvable_df[wer_col].dropna()
                    if len(pipeline_wer) > 0:
                        friendly_name = get_friendly_pipeline_name(pipeline)
                        fig.add_trace(go.Histogram(
                            x=pipeline_wer,
                            name=friendly_name,
                            opacity=0.6,
                            nbinsx=20,
                            marker_color=colors[i % len(colors)]
                        ))
            
            fig.update_layout(
                title="WER Distribution: Improvable Segments Only (Baseline WER > 0)",
                xaxis_title="Word Error Rate",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed statistics table
            st.subheader("📋 Performance Statistics: Improvable Segments Only")
            
            display_df = improvable_stats_df.copy()
            for col in ['Improved_%', 'Same_%', 'Degraded_%', 'Relative_Reduction_%']:
                display_df[col] = display_df[col].round(1)
            for col in ['Baseline_WER', 'Pipeline_WER', 'WER_Reduction']:
                display_df[col] = display_df[col].round(4)
            
            st.dataframe(
                display_df[['Pipeline', 'Total_Segments', 'Improved_%', 'Same_%', 'Degraded_%', 
                           'Baseline_WER', 'Pipeline_WER', 'WER_Reduction', 'Relative_Reduction_%']],
                use_container_width=True
            )
            
            st.success(f"""
            **Real Impact Summary:** When focusing on the {len(improvable_df)} segments where improvement 
            is actually possible (excluding perfect transcriptions), the pipelines show their true effectiveness.
            """)
    else:
        st.warning("No improvable segments found (all baseline WER = 0).")
    
    st.markdown("---")
    
    # WER improvement distribution
    st.subheader("📊 WER Improvement Distribution")
    
    improvement_data = []
    
    for pipeline in pipelines:
        wer_col = f'{pipeline}_wer'
        if wer_col in df.columns:
            valid_mask = (~df['stt_raw_wer'].isna()) & (~df[wer_col].isna())
            valid_df = df[valid_mask]
            
            if len(valid_df) > 0:
                improvements = valid_df['stt_raw_wer'] - valid_df[wer_col]
                friendly_name = get_friendly_pipeline_name(pipeline)
                improvement_data.extend([{
                    'Pipeline': friendly_name,
                    'WER_Improvement': imp,
                    'Absolute_Improvement': abs(imp),
                    'Improvement_Category': 'Improved' if imp > 1e-6 else ('Degraded' if imp < -1e-6 else 'Same')
                } for imp in improvements])
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Box plot of improvements
    fig = px.box(
        improvement_df, 
        x='Pipeline', 
        y='WER_Improvement',
        title="WER Improvement Distribution by Pipeline",
        points="outliers",
        color='Pipeline',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No Change")
    fig.update_layout(
        yaxis_title="WER Improvement (Baseline - Pipeline)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics table
    st.subheader("📈 Statistical Summary")
    
    summary_stats = improvement_df.groupby('Pipeline')['WER_Improvement'].agg([
        'count', 'mean', 'std', 'min', 'max', 
        lambda x: x.quantile(0.25), 
        lambda x: x.quantile(0.5), 
        lambda x: x.quantile(0.75)
    ]).round(4)
    
    summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Q1', 'Median', 'Q3']
    
    st.dataframe(summary_stats, use_container_width=True)

if __name__ == "__main__":
    main() 
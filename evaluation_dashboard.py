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
                    'Pipeline': pipeline.replace('Pipeline', '').replace('Generate', '').replace('Whisper', 'Whisper '),
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
        pipeline_name = best_pipeline['Pipeline'].replace('FixTranscriptByLLM', 'Fix LLM').replace('Generate', '').replace('Whisper ', 'Whisper')
        st.metric(
            label="Best Pipeline",
            value=pipeline_name,
            delta=f"{best_pipeline['Improved_%']:.1f}% improvement rate",
            delta_color="normal"
        )
    
    with col3:
        safest_pipeline = stats_df.loc[stats_df['Degraded_%'].idxmin()]
        safe_pipeline_name = safest_pipeline['Pipeline'].replace('FixTranscriptByLLM', 'Fix LLM').replace('Generate', '').replace('Whisper ', 'Whisper')
        st.metric(
            label="Safest Pipeline",
            value=safe_pipeline_name,
            delta=f"{safest_pipeline['Degraded_%']:.1f}% degradation risk",
            delta_color="inverse"
        )
    
    with col4:
        avg_baseline_wer = df['stt_raw_wer'].mean()
        st.metric(
            label="Baseline WER",
            value=f"{avg_baseline_wer:.3f}",
            help="Average Word Error Rate across all segments"
        )
    
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
    
    # Pipeline selection
    pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                'GenerateNamesPipeline', 'GenerateTopicPipeline']
    
    selected_pipelines = st.multiselect(
        "Select Pipelines to Compare",
        pipelines,
        default=pipelines[:2]
    )
    
    if len(selected_pipelines) < 2:
        st.warning("Please select at least 2 pipelines for comparison.")
        return
    
    # WER distribution comparison
    st.subheader("WER Distribution Comparison")
    
    fig = go.Figure()
    
    for pipeline in selected_pipelines:
        wer_col = f'{pipeline}_wer'
        if wer_col in df.columns:
            valid_data = df[df[wer_col].notna()][wer_col]
            fig.add_trace(go.Histogram(
                x=valid_data,
                name=pipeline.replace('Pipeline', ''),
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
            st.metric(f"{pipeline1} Wins", pipeline1_wins)
        
        with col2:
            st.metric("Ties", ties)
        
        with col3:
            st.metric(f"{pipeline2} Wins", pipeline2_wins)
        
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
            hovertemplate=f"{pipeline1}: %{{x:.3f}}<br>{pipeline2}: %{{y:.3f}}<br>Video: %{{text}}<extra></extra>"
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
            title=f"{pipeline1} vs {pipeline2} Performance",
            xaxis_title=f"{pipeline1} WER",
            yaxis_title=f"{pipeline2} WER"
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
        st.markdown(f'<div class="transcript-box ground-truth-box">{sample["transcript"]}</div>', unsafe_allow_html=True)
        
        # Baseline (raw STT)
        st.subheader("📄 Baseline (Raw Whisper)")
        baseline_text = ""
        if 'stt_raw_norm' in sample and pd.notna(sample['stt_raw_norm']):
            baseline_text = sample['stt_raw_norm']
            ground_truth = sample['transcript'].lower().strip()
            
            # Highlight differences from ground truth
            gt_highlighted, baseline_highlighted = highlight_differences(ground_truth, baseline_text)
            st.markdown(f'<div class="transcript-box baseline-box">{baseline_highlighted}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="transcript-box baseline-box"><em>Baseline transcription not available</em></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # All Pipeline Results
        st.subheader("🔄 All Pipeline Results")
        
        pipelines = ['FixTranscriptByLLMPipeline', 'GenerateWhisperPromptPipeline', 
                    'GenerateNamesPipeline', 'GenerateTopicPipeline']
        
        for pipeline in pipelines:
            pipeline_display_name = pipeline.replace('Pipeline', '').replace('Generate', '').replace('Whisper', 'Whisper ')
            
            with st.expander(f"🔧 {pipeline_display_name}", expanded=True):
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
                    # Pipeline result text
                    pipeline_norm_col = f'{pipeline}_norm'
                    if pipeline_norm_col in sample and pd.notna(sample[pipeline_norm_col]):
                        pipeline_text = sample[pipeline_norm_col]
                        
                        # Highlight differences from baseline
                        if baseline_text:
                            baseline_highlighted2, pipeline_highlighted = highlight_differences(baseline_text, pipeline_text)
                            st.markdown(f'<div class="transcript-box pipeline-box">{pipeline_highlighted}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="transcript-box pipeline-box">{pipeline_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="transcript-box pipeline-box"><em>No transcription available</em></div>', unsafe_allow_html=True)
                
                # Show initial prompt if available
                pipeline_prompt_col = f'{pipeline}_initial_prompt'
                if pipeline_prompt_col in sample and pd.notna(sample[pipeline_prompt_col]):
                    st.markdown("**🎯 Generated Prompt:**")
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
                    video_stat[f'{pipeline}_Improvement_%'] = (improved / total) * 100
                    video_stat[f'{pipeline}_Baseline_WER'] = valid_df['stt_raw_wer'].mean()
                    video_stat[f'{pipeline}_Pipeline_WER'] = valid_df[wer_col].mean()
        
        video_stats.append(video_stat)
    
    video_stats_df = pd.DataFrame(video_stats)
    
    # Video performance heatmap
    st.subheader("Improvement Rate by Video")
    
    improvement_cols = [col for col in video_stats_df.columns if col.endswith('_Improvement_%')]
    
    if improvement_cols:
        heatmap_data = video_stats_df.set_index('Video_ID')[improvement_cols]
        heatmap_data.columns = [col.replace('_Improvement_%', '').replace('Pipeline', '') for col in heatmap_data.columns]
        
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
        improvement_col = f'{pipeline}_Improvement_%'
        baseline_col = f'{pipeline}_Baseline_WER'
        
        if improvement_col in video_stats_df.columns and baseline_col in video_stats_df.columns:
            fig.add_trace(go.Scatter(
                x=video_stats_df[baseline_col],
                y=video_stats_df[improvement_col],
                mode='markers+text',
                text=video_stats_df['Video_ID'],
                textposition="top center",
                name=pipeline.replace('Pipeline', ''),
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
                improvement_data.extend([{
                    'Pipeline': pipeline.replace('Pipeline', '').replace('Generate', '').replace('Whisper', 'Whisper '),
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
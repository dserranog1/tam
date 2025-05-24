import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(
    page_title="ML Models Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        # Load datasets
        dataset = pd.read_csv("assets/ames_dataset.csv")
        metrics = pd.read_csv("assets/model_metrics.csv")
        predictions = pd.read_csv("assets/model_predictions.csv")
        return dataset, metrics, predictions
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def calculate_additional_metrics(y_true, y_pred):
    """Calculate additional metrics for model evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(height=600)
    return fig

def create_model_comparison_chart(metrics_df):
    """Create model comparison bar chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean Absolute Error (Lower is Better)', 
                       'Root Mean Square Error (Lower is Better)',
                       'R² Score (Higher is Better)', 
                       'Mean Absolute Percentage Error (Lower is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # MAE
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'], 
               name='MAE', marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], 
               name='RMSE', marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # R²
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['R2'], 
               name='R²', marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # MAPE
    fig.add_trace(
        go.Bar(x=metrics_df['Model'], y=metrics_df['MAPE'], 
               name='MAPE', marker_color=colors, showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Model Performance Comparison")
    return fig

def create_prediction_scatter_plots(predictions_df):
    """Create scatter plots for predictions vs actual"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Random Forest', 'SVR', 'Kernel Ridge'),
        shared_yaxes=True
    )
    
    models = ['y_pred_rf', 'y_pred_svr', 'y_pred_kernel']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (model, color) in enumerate(zip(models, colors), 1):
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=predictions_df['y_true'], 
                y=predictions_df[model],
                mode='markers',
                name=model.replace('y_pred_', '').upper(),
                marker=dict(color=color, size=6, opacity=0.6),
                showlegend=False
            ),
            row=1, col=i
        )
        
        # Perfect prediction line
        min_val = min(predictions_df['y_true'].min(), predictions_df[model].min())
        max_val = max(predictions_df['y_true'].max(), predictions_df[model].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction',
                showlegend=i==1
            ),
            row=1, col=i
        )
    
    fig.update_xaxes(title_text="Actual Values")
    fig.update_yaxes(title_text="Predicted Values", col=1)
    fig.update_layout(height=500, title_text="Predictions vs Actual Values")
    
    return fig

def create_residuals_plot(predictions_df):
    """Create residuals plot for all models"""
    fig = go.Figure()
    
    models = ['y_pred_rf', 'y_pred_svr', 'y_pred_kernel']
    model_names = ['Random Forest', 'SVR', 'Kernel Ridge']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for model, name, color in zip(models, model_names, colors):
        residuals = predictions_df['y_true'] - predictions_df[model]
        
        fig.add_trace(
            go.Scatter(
                x=predictions_df[model],
                y=residuals,
                mode='markers',
                name=name,
                marker=dict(color=color, size=6, opacity=0.6)
            )
        )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Perfect Prediction")
    
    fig.update_layout(
        title="Residuals Plot (Actual - Predicted)",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=500
    )
    
    return fig

def main():
    # Load data
    dataset, metrics_df, predictions_df = load_data()
    
    # Title
    st.markdown('<h1 class="main-header">🎯 ML Models Performance Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Comparing Random Forest, SVR, and Kernel Ridge Regression Models")
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Overview", "🔍 Dataset Analysis", "📈 Model Comparison", "🎯 Predictions Analysis"]
    )
    
    if page == "📊 Overview":
        st.markdown('<h2 class="section-header">📊 Project Overview</h2>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Size", f"{len(dataset):,} samples")
            st.metric("Features", f"{len(dataset.columns)} columns")
        
        with col2:
            st.metric("Models Trained", "3")
            st.metric("Best R² Score", f"{metrics_df['R2'].max():.4f}")
        
        with col3:
            best_model = metrics_df.loc[metrics_df['R2'].idxmax(), 'Model']
            st.metric("Best Model", best_model)
            st.metric("Lowest RMSE", f"{metrics_df['RMSE'].min():.4f}")
        
        # Quick metrics comparison
        st.markdown('<h3 class="section-header">📋 Quick Metrics Summary</h3>', 
                    unsafe_allow_html=True)
        
        # Style the metrics dataframe
        styled_metrics = metrics_df.style.format({
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'R2': '{:.4f}',
            'MAPE': '{:.2f}%'
        }).highlight_max(subset=['R2'], color='lightgreen')\
          .highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen')
        
        st.dataframe(styled_metrics, use_container_width=True)
        
        # Model ranking
        st.markdown('<h3 class="section-header">🏆 Model Ranking</h3>', 
                    unsafe_allow_html=True)
        
        # Rank by R² score (higher is better)
        ranking_df = metrics_df.sort_values('R2', ascending=False).reset_index(drop=True)
        ranking_df.index = ranking_df.index + 1
        st.dataframe(ranking_df, use_container_width=True)
    
    elif page == "🔍 Dataset Analysis":
        st.markdown('<h2 class="section-header">🔍 Dataset Analysis</h2>', 
                    unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📋 Basic Info", "📊 Distributions", "🔗 Correlations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Dataset Shape")
                st.write(f"**Rows:** {dataset.shape[0]:,}")
                st.write(f"**Columns:** {dataset.shape[1]:,}")
                
                st.markdown("#### Data Types")
                dtype_counts = dataset.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"**{dtype}:** {count} columns")
            
            with col2:
                st.markdown("#### Missing Values")
                missing_data = dataset.isnull().sum()
                missing_percent = (missing_data / len(dataset)) * 100
                
                if missing_data.sum() == 0:
                    st.success("No missing values found! 🎉")
                else:
                    missing_df = pd.DataFrame({
                        'Missing Count': missing_data[missing_data > 0],
                        'Percentage': missing_percent[missing_data > 0]
                    })
                    st.dataframe(missing_df)
            
            st.markdown("#### Statistical Summary")
            st.dataframe(dataset.describe(), use_container_width=True)
        
        with tab2:
            st.markdown("#### Feature Distributions")
            
            # Select numeric columns for distribution plots
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                selected_features = st.multiselect(
                    "Select features to visualize",
                    numeric_cols,
                    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                )
                
                if selected_features:
                    cols = min(2, len(selected_features))
                    rows = (len(selected_features) + 1) // 2
                    
                    fig = make_subplots(
                        rows=rows, cols=cols,
                        subplot_titles=selected_features
                    )
                    
                    for i, feature in enumerate(selected_features):
                        row = i // cols + 1
                        col = i % cols + 1
                        
                        fig.add_trace(
                            go.Histogram(x=dataset[feature], name=feature, showlegend=False),
                            row=row, col=col
                        )
                    
                    fig.update_layout(height=300*rows, title_text="Feature Distributions")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for distribution plots.")
        
        with tab3:
            st.markdown("#### Correlation Analysis")
            
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_fig = create_correlation_heatmap(dataset)
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                # Show highest correlations
                corr_matrix = dataset[numeric_cols].corr()
                
                # Get upper triangle of correlation matrix
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Find pairs with high correlation
                high_corr_pairs = []
                for col in upper_tri.columns:
                    for idx in upper_tri.index:
                        if abs(upper_tri.loc[idx, col]) > 0.7:
                            high_corr_pairs.append({
                                'Feature 1': idx,
                                'Feature 2': col,
                                'Correlation': upper_tri.loc[idx, col]
                            })
                
                if high_corr_pairs:
                    st.markdown("#### High Correlations (|r| > 0.7)")
                    high_corr_df = pd.DataFrame(high_corr_pairs)
                    st.dataframe(high_corr_df.sort_values('Correlation', key=abs, ascending=False))
                else:
                    st.info("No high correlations (|r| > 0.7) found between features.")
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
    
    elif page == "📈 Model Comparison":
        st.markdown('<h2 class="section-header">📈 Model Performance Comparison</h2>', 
                    unsafe_allow_html=True)
        
        # Metrics comparison chart
        comparison_fig = create_model_comparison_chart(metrics_df)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("#### Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Performance Metrics")
            styled_metrics = metrics_df.set_index('Model').style.format({
                'MAE': '{:.6f}',
                'RMSE': '{:.6f}',
                'R2': '{:.6f}',
                'MAPE': '{:.4f}%'
            }).highlight_max(subset=['R2'], color='lightgreen')\
              .highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen')
            
            st.dataframe(styled_metrics)
        
        with col2:
            st.markdown("##### Model Insights")
            
            best_r2_model = metrics_df.loc[metrics_df['R2'].idxmax()]
            best_mae_model = metrics_df.loc[metrics_df['MAE'].idxmin()]
            best_rmse_model = metrics_df.loc[metrics_df['RMSE'].idxmin()]
            
            st.markdown(f"**🏆 Highest R² Score:** {best_r2_model['Model']} ({best_r2_model['R2']:.4f})")
            st.markdown(f"**📉 Lowest MAE:** {best_mae_model['Model']} ({best_mae_model['MAE']:.4f})")
            st.markdown(f"**📊 Lowest RMSE:** {best_rmse_model['Model']} ({best_rmse_model['RMSE']:.4f})")
            
            # Performance ranking
            st.markdown("##### Overall Ranking")
            
            # Calculate ranks for each metric
            mae_rank = metrics_df['MAE'].rank(ascending=True)
            rmse_rank = metrics_df['RMSE'].rank(ascending=True)
            mape_rank = metrics_df['MAPE'].rank(ascending=True)
            r2_rank = metrics_df['R2'].rank(ascending=False)
            
            # Calculate average rank for each model
            metrics_df_copy = metrics_df.copy()
            metrics_df_copy['Overall_Rank'] = (mae_rank + rmse_rank + mape_rank + r2_rank) / 4
            
            ranked_models = metrics_df_copy.sort_values('Overall_Rank')[['Model', 'Overall_Rank']]
            
            for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
                st.markdown(f"**{i}.** {row['Model']}")
    
    elif page == "🎯 Predictions Analysis":
        st.markdown('<h2 class="section-header">🎯 Predictions Analysis</h2>', 
                    unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Predictions vs Actual", "📈 Residuals Analysis", "🔍 Error Distribution"])
        
        with tab1:
            st.markdown("#### Predictions vs Actual Values")
            scatter_fig = create_prediction_scatter_plots(predictions_df)
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Points closer to the red dashed line indicate better predictions
            - The red line represents perfect predictions (predicted = actual)
            - Scatter around the line shows prediction variability
            """)
        
        with tab2:
            st.markdown("#### Residuals Analysis")
            residuals_fig = create_residuals_plot(predictions_df)
            st.plotly_chart(residuals_fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Residuals = Actual - Predicted values
            - Points closer to the horizontal line (y=0) indicate better predictions
            - Random scatter around y=0 suggests good model fit
            - Patterns in residuals may indicate model bias
            """)
        
        with tab3:
            st.markdown("#### Error Distribution")
            
            # Calculate errors for each model
            models = ['y_pred_rf', 'y_pred_svr', 'y_pred_kernel']
            model_names = ['Random Forest', 'SVR', 'Kernel Ridge']
            
            fig = go.Figure()
            
            for model, name in zip(models, model_names):
                errors = predictions_df['y_true'] - predictions_df[model]
                
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        name=name,
                        opacity=0.7,
                        nbinsx=30
                    )
                )
            
            fig.update_layout(
                title="Distribution of Prediction Errors",
                xaxis_title="Error (Actual - Predicted)",
                yaxis_title="Frequency",
                barmode='overlay',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error statistics
            st.markdown("#### Error Statistics")
            
            error_stats = []
            for model, name in zip(models, model_names):
                errors = predictions_df['y_true'] - predictions_df[model]
                stats = {
                    'Model': name,
                    'Mean Error': errors.mean(),
                    'Std Error': errors.std(),
                    'Min Error': errors.min(),
                    'Max Error': errors.max(),
                    'Mean Abs Error': abs(errors).mean()
                }
                error_stats.append(stats)
            
            error_df = pd.DataFrame(error_stats)
            st.dataframe(error_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Dashboard created for ML Model Comparison** | "
        "Models: Random Forest, SVR, Kernel Ridge"
    )

if __name__ == "__main__":
    main()
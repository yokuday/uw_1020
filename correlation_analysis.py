import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.contingency_tables import mcnemar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import itertools
from collections import defaultdict

class SteamAnalysisEngine:
    def __init__(self, data_path=None, dataframe=None):
        self.data_path = data_path
        self.df = dataframe if dataframe is not None else pd.read_csv(data_path)
        self.processed_df = None
        self.feature_columns = []
        self.target_columns = []
        self.categorical_features = []
        self.numerical_features = []
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.encoders = {}
        
        self.setup_analysis_environment()
        self.validate_data_quality()
        self.preprocess_data()
        
    def setup_analysis_environment(self):
        warnings.filterwarnings('ignore', category=FutureWarning)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff871f',
            'info': '#17a2b8',
            'light': '#e9ecef',
            'dark': '#343a40'
        }
        
    def validate_data_quality(self):
        data_quality_report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.value_counts().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns),
            'date_columns': list(self.df.select_dtypes(include=['datetime64']).columns)
        }
        
        outlier_detection = {}
        for col in data_quality_report['numeric_columns']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            outlier_detection[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        data_quality_report['outliers'] = outlier_detection
        self.data_quality = data_quality_report
        
    def preprocess_data(self):
        self.processed_df = self.df.copy()
        
        if 'release_date' in self.processed_df.columns:
            self.processed_df['release_date'] = pd.to_datetime(self.processed_df['release_date'], errors='coerce')
            self.processed_df['release_year'] = self.processed_df['release_date'].dt.year
            self.processed_df['release_month'] = self.processed_df['release_date'].dt.month
            self.processed_df['release_quarter'] = self.processed_df['release_date'].dt.quarter
            self.processed_df['days_since_release'] = (datetime.now() - self.processed_df['release_date']).dt.days
            
        if 'demo_release_date' in self.processed_df.columns:
            self.processed_df['demo_release_date'] = pd.to_datetime(self.processed_df['demo_release_date'], errors='coerce')
            self.processed_df['demo_to_release_gap'] = (self.processed_df['release_date'] - self.processed_df['demo_release_date']).dt.days
            
        if 'tags' in self.processed_df.columns:
            self.processed_df['tag_count'] = self.processed_df['tags'].str.count(',') + 1
            popular_tags = ['Action', 'Adventure', 'Casual', 'Indie', 'Simulation', 'Strategy', 'RPG']
            for tag in popular_tags:
                self.processed_df[f'has_{tag.lower()}_tag'] = self.processed_df['tags'].str.contains(tag, case=False, na=False)
                
        if 'price' in self.processed_df.columns:
            self.processed_df['price_category'] = pd.cut(self.processed_df['price'], 
                                                        bins=[0, 5, 15, 30, 50, float('inf')], 
                                                        labels=['Free/Cheap', 'Budget', 'Mid-range', 'Premium', 'Expensive'])
                                                        
        if 'revenue' in self.processed_df.columns:
            self.processed_df['log_revenue'] = np.log1p(self.processed_df['revenue'].fillna(0))
            self.processed_df['revenue_category'] = pd.cut(self.processed_df['revenue'], 
                                                          bins=[0, 1000, 10000, 100000, 1000000, float('inf')], 
                                                          labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                                                          
        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['tags', 'system_requirements', 'supported_languages']:
                le = LabelEncoder()
                self.processed_df[f'{col}_encoded'] = le.fit_transform(self.processed_df[col].fillna('Unknown'))
                self.encoders[col] = le
                
        self.numerical_features = list(self.processed_df.select_dtypes(include=[np.number]).columns)
        self.categorical_features = [col for col in categorical_cols if col not in self.encoders.keys()]
        
        missing_threshold = 0.3
        columns_to_drop = []
        for col in self.processed_df.columns:
            missing_ratio = self.processed_df[col].isnull().sum() / len(self.processed_df)
            if missing_ratio > missing_threshold:
                columns_to_drop.append(col)
                
        self.processed_df = self.processed_df.drop(columns=columns_to_drop)
        
        for col in self.numerical_features:
            if col in self.processed_df.columns:
                median_value = self.processed_df[col].median()
                self.processed_df[col] = self.processed_df[col].fillna(median_value)
                
    def analyze_demo_correlation(self, target_variable='revenue'):
        correlation_results = {}
        
        numeric_cols = [col for col in self.numerical_features if col in self.processed_df.columns]
        correlation_matrix = self.processed_df[numeric_cols].corr(method='pearson')
        
        spearman_matrix = self.processed_df[numeric_cols].corr(method='spearman')
        kendall_matrix = self.processed_df[numeric_cols].corr(method='kendall')
        
        correlation_results['pearson'] = correlation_matrix
        correlation_results['spearman'] = spearman_matrix
        correlation_results['kendall'] = kendall_matrix
        
        p_values_matrix = np.zeros((len(numeric_cols), len(numeric_cols)))
        confidence_intervals = {}
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i != j:
                    data1 = self.processed_df[col1].dropna()
                    data2 = self.processed_df[col2].dropna()
                    
                    if len(data1) > 10 and len(data2) > 10:
                        corr, p_val = stats.pearsonr(data1, data2)
                        p_values_matrix[i][j] = p_val
                        
                        n = min(len(data1), len(data2))
                        z_transform = np.arctanh(corr)
                        standard_error = 1 / np.sqrt(n - 3)
                        ci_lower = np.tanh(z_transform - 1.96 * standard_error)
                        ci_upper = np.tanh(z_transform + 1.96 * standard_error)
                        confidence_intervals[f'{col1}_vs_{col2}'] = (ci_lower, ci_upper)
                        
        correlation_results['p_values'] = pd.DataFrame(p_values_matrix, 
                                                      index=numeric_cols, 
                                                      columns=numeric_cols)
        correlation_results['confidence_intervals'] = confidence_intervals
        
        demo_specific_correlations = {}
        demo_columns = [col for col in numeric_cols if 'demo' in col.lower()]
        
        for demo_col in demo_columns:
            for target_col in [target_variable, 'wishlist_count', 'review_score']:
                if target_col in self.processed_df.columns:
                    valid_data = self.processed_df[[demo_col, target_col]].dropna()
                    if len(valid_data) > 30:
                        corr, p_val = stats.pearsonr(valid_data[demo_col], valid_data[target_col])
                        demo_specific_correlations[f'{demo_col}_vs_{target_col}'] = {
                            'correlation': corr,
                            'p_value': p_val,
                            'sample_size': len(valid_data),
                            'significant': p_val < 0.05
                        }
                        
        correlation_results['demo_specific'] = demo_specific_correlations
        self.results['correlation_analysis'] = correlation_results
        
        return correlation_results
        
    def revenue_comparison_analysis(self, groupby_column='demo_available'):
        comparison_results = {}
        
        if groupby_column not in self.processed_df.columns:
            raise ValueError(f"Column '{groupby_column}' not found in dataset")
            
        groups = self.processed_df.groupby(groupby_column)['revenue'].apply(list)
        group_names = list(groups.index)
        group_data = list(groups.values)
        
        descriptive_stats = {}
        for name, data in zip(group_names, group_data):
            data_clean = [x for x in data if pd.notna(x) and x > 0]
            if len(data_clean) > 0:
                descriptive_stats[name] = {
                    'count': len(data_clean),
                    'mean': np.mean(data_clean),
                    'median': np.median(data_clean),
                    'std': np.std(data_clean),
                    'min': np.min(data_clean),
                    'max': np.max(data_clean),
                    'q25': np.percentile(data_clean, 25),
                    'q75': np.percentile(data_clean, 75),
                    'iqr': np.percentile(data_clean, 75) - np.percentile(data_clean, 25),
                    'skewness': stats.skew(data_clean),
                    'kurtosis': stats.kurtosis(data_clean)
                }
                
        comparison_results['descriptive_stats'] = descriptive_stats
        
        if len(group_data) == 2:
            group1, group2 = [pd.Series(data).dropna() for data in group_data]
            
            statistic, p_value_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            comparison_results['mann_whitney_u'] = {
                'statistic': statistic,
                'p_value': p_value_mw,
                'significant': p_value_mw < 0.05
            }
            
            try:
                statistic_ttest, p_value_ttest = stats.ttest_ind(group1, group2, equal_var=False)
                comparison_results['welch_t_test'] = {
                    'statistic': statistic_ttest,
                    'p_value': p_value_ttest,
                    'significant': p_value_ttest < 0.05
                }
            except Exception:
                pass
                
            try:
                statistic_ks, p_value_ks = stats.ks_2samp(group1, group2)
                comparison_results['kolmogorov_smirnov'] = {
                    'statistic': statistic_ks,
                    'p_value': p_value_ks,
                    'significant': p_value_ks < 0.05
                }
            except Exception:
                pass
                
        elif len(group_data) > 2:
            clean_groups = [pd.Series(data).dropna() for data in group_data if len(pd.Series(data).dropna()) > 0]
            
            if len(clean_groups) > 2:
                try:
                    statistic_kw, p_value_kw = stats.kruskal(*clean_groups)
                    comparison_results['kruskal_wallis'] = {
                        'statistic': statistic_kw,
                        'p_value': p_value_kw,
                        'significant': p_value_kw < 0.05
                    }
                except Exception:
                    pass
                    
                try:
                    statistic_anova, p_value_anova = stats.f_oneway(*clean_groups)
                    comparison_results['anova'] = {
                        'statistic': statistic_anova,
                        'p_value': p_value_anova,
                        'significant': p_value_anova < 0.05
                    }
                except Exception:
                    pass
                    
        effect_sizes = {}
        if len(group_data) == 2:
            group1, group2 = [pd.Series(data).dropna() for data in group_data]
            if len(group1) > 0 and len(group2) > 0:
                pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var()) / 
                                   (len(group1) + len(group2) - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_std
                effect_sizes['cohens_d'] = cohens_d
                
                u_statistic = comparison_results.get('mann_whitney_u', {}).get('statistic', 0)
                n1, n2 = len(group1), len(group2)
                rank_biserial = 1 - (2 * u_statistic) / (n1 * n2)
                effect_sizes['rank_biserial'] = rank_biserial
                
        comparison_results['effect_sizes'] = effect_sizes
        
        percentage_improvements = {}
        if len(descriptive_stats) >= 2:
            stats_list = list(descriptive_stats.items())
            for i in range(len(stats_list)):
                for j in range(i + 1, len(stats_list)):
                    name1, stats1 = stats_list[i]
                    name2, stats2 = stats_list[j]
                    
                    median_improvement = ((stats1['median'] - stats2['median']) / stats2['median']) * 100
                    mean_improvement = ((stats1['mean'] - stats2['mean']) / stats2['mean']) * 100
                    
                    percentage_improvements[f'{name1}_vs_{name2}'] = {
                        'median_improvement': median_improvement,
                        'mean_improvement': mean_improvement
                    }
                    
        comparison_results['percentage_improvements'] = percentage_improvements
        self.results['revenue_comparison'] = comparison_results
        
        return comparison_results

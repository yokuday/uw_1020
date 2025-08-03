import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
import sqlite3
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
from scipy import stats
from scipy.stats import zscore
import missingno as msno
from dataprep.eda import create_report
from pandas_profiling import ProfileReport
import logging
from typing import Dict, List, Tuple, Optional, Union
import pickle
import yaml

class SteamDataProcessor:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path) if config_path else self.default_config()
        self.processing_log = []
        self.data_quality_metrics = {}
        self.transformation_history = []
        self.feature_importance_scores = {}
        self.outlier_detection_results = {}
        self.imputation_strategies = {}
        self.scaling_parameters = {}
        self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.log_processing_step(f"Error loading config: {str(e)}", "WARNING")
            return self.default_config()
            
    def default_config(self) -> Dict:
        return {
            'missing_data_threshold': 0.3,
            'outlier_detection_method': 'iqr',
            'outlier_threshold': 1.5,
            'scaling_method': 'robust',
            'imputation_method': 'knn',
            'feature_selection_k': 20,
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95,
            'sample_size_minimum': 100,
            'data_types': {
                'app_id': 'int64',
                'revenue': 'float64',
                'price': 'float64',
                'demo_score': 'float64',
                'wishlist_count': 'int64',
                'release_date': 'datetime64[ns]',
                'demo_available': 'bool'
            }
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_processing_step(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'level': level
        }
        self.processing_log.append(log_entry)
        
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
            
    def clean_steam_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.log_processing_step("Starting comprehensive data cleaning pipeline")
        
        df = raw_df.copy()
        initial_shape = df.shape
        self.log_processing_step(f"Initial dataset shape: {initial_shape}")
        
        df = self.remove_duplicates(df)
        df = self.standardize_data_types(df)
        df = self.handle_missing_values(df)
        df = self.filter_date_range(df)
        df = self.filter_indie_games(df)
        df = self.validate_numeric_ranges(df)
        df = self.clean_text_fields(df)
        df = self.create_derived_features(df)
        df = self.detect_and_handle_outliers(df)
        df = self.validate_data_consistency(df)
        
        final_shape = df.shape
        self.log_processing_step(f"Final dataset shape: {final_shape}")
        self.log_processing_step(f"Removed {initial_shape[0] - final_shape[0]} rows and {initial_shape[1] - final_shape[1]} columns")
        
        self.generate_data_quality_report(df)
        return df
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        
        if 'app_id' in df.columns:
            df = df.drop_duplicates(subset=['app_id'], keep='first')
            self.log_processing_step(f"Removed {initial_count - len(df)} duplicate app_ids")
        else:
            df = df.drop_duplicates()
            self.log_processing_step(f"Removed {initial_count - len(df)} duplicate rows")
            
        duplicate_check = df.duplicated().sum()
        if duplicate_check > 0:
            self.log_processing_step(f"Warning: {duplicate_check} duplicates remain", "WARNING")
            
        return df
        
    def standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_processing_step("Standardizing data types")
        
        for column, dtype in self.config['data_types'].items():
            if column in df.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif dtype == 'bool':
                        df[column] = df[column].astype(bool)
                    elif dtype in ['int64', 'float64']:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        if dtype == 'int64':
                            df[column] = df[column].astype('Int64')
                    else:
                        df[column] = df[column].astype(dtype)
                        
                    self.log_processing_step(f"Converted {column} to {dtype}")
                except Exception as e:
                    self.log_processing_step(f"Failed to convert {column} to {dtype}: {str(e)}", "WARNING")
                    
        price_columns = ['price', 'original_price', 'discount_price']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].replace('Free', 0)
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        percentage_columns = ['discount_percent', 'demo_score', 'positive_rating']
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(0, 100)
                
        return df
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_processing_step("Handling missing values")
        
        missing_summary = df.isnull().sum()
        missing_percentages = (missing_summary / len(df)) * 100
        
        columns_to_drop = []
        threshold = self.config['missing_data_threshold'] * 100
        
        for column, percentage in missing_percentages.items():
            if percentage > threshold:
                columns_to_drop.append(column)
                self.log_processing_step(f"Marking {column} for removal ({percentage:.2f}% missing)")
                
        df = df.drop(columns=columns_to_drop)
        self.log_processing_step(f"Dropped {len(columns_to_drop)} columns due to excessive missing values")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if df[column].isnull().sum() > 0:
                if column in ['price', 'original_price']:
                    median_price = df[column].median()
                    df[column] = df[column].fillna(median_price)
                elif column in ['wishlist_count', 'followers', 'review_count']:
                    df[column] = df[column].fillna(0)
                elif column in ['demo_score', 'positive_rating']:
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        df[column] = df[column].fillna(mode_value[0])
                    else:
                        df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(df[column].median())
                    
                self.log_processing_step(f"Filled missing values in {column}")
                
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            if df[column].isnull().sum() > 0:
                if column in ['developer', 'publisher']:
                    df[column] = df[column].fillna('Unknown')
                elif column in ['tags', 'genres']:
                    df[column] = df[column].fillna('Untagged')
                else:
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        df[column] = df[column].fillna(mode_value[0])
                    else:
                        df[column] = df[column].fillna('Unknown')
                        
                self.log_processing_step(f"Filled missing values in {column}")
                
        boolean_columns = df.select_dtypes(include=['bool']).columns
        for column in boolean_columns:
            if df[column].isnull().sum() > 0:
                df[column] = df[column].fillna(False)
                self.log_processing_step(f"Filled missing boolean values in {column}")
                
        return df
        
    def filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'release_date' not in df.columns:
            return df
            
        df = df[df['release_date'].notna()]
        
        start_date = pd.to_datetime('2019-01-01')
        end_date = pd.to_datetime('2024-12-31')
        
        initial_count = len(df)
        df = df[(df['release_date'] >= start_date) & (df['release_date'] <= end_date)]
        filtered_count = initial_count - len(df)
        
        self.log_processing_step(f"Filtered {filtered_count} games outside 2019-2024 date range")
        
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_quarter'] = df['release_date'].dt.quarter
        df['release_day_of_week'] = df['release_date'].dt.dayofweek
        df['days_since_release'] = (datetime.now() - df['release_date']).dt.days
        
        seasonal_mapping = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                           3: 'Spring', 4: 'Spring', 5: 'Spring',
                           6: 'Summer', 7: 'Summer', 8: 'Summer',
                           9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['release_season'] = df['release_month'].map(seasonal_mapping)
        
        return df
        
    def filter_indie_games(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        
        indie_filters = []
        
        if 'tags' in df.columns:
            indie_filters.append(df['tags'].str.contains('Indie', case=False, na=False))
            
        if 'genres' in df.columns:
            indie_filters.append(df['genres'].str.contains('Indie', case=False, na=False))
            
        if 'categories' in df.columns:
            indie_filters.append(df['categories'].str.contains('Indie', case=False, na=False))
            
        if indie_filters:
            combined_filter = indie_filters[0]
            for filter_condition in indie_filters[1:]:
                combined_filter = combined_filter | filter_condition
            df = df[combined_filter]
            
        filtered_count = initial_count - len(df)
        self.log_processing_step(f"Filtered to indie games only, removed {filtered_count} non-indie games")
        
        if 'developer' in df.columns:
            major_publishers = ['Electronic Arts', 'Activision', 'Ubisoft', 'Take-Two Interactive', 
                              'Sony Interactive Entertainment', 'Microsoft Studios', 'Nintendo']
            for publisher in major_publishers:
                initial_len = len(df)
                df = df[~df['developer'].str.contains(publisher, case=False, na=False)]
                removed_count = initial_len - len(df)
                if removed_count > 0:
                    self.log_processing_step(f"Removed {removed_count} games from major publisher {publisher}")
                    
        price_threshold = 60
        if 'price' in df.columns:
            expensive_games = len(df[df['price'] > price_threshold])
            if expensive_games > 0:
                self.log_processing_step(f"Found {expensive_games} games priced above ${price_threshold} (keeping as premium indie)")
                
        return df
        
    def validate_numeric_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        validation_rules = {
            'price': (0, 1000),
            'demo_score': (0, 100),
            'positive_rating': (0, 100),
            'discount_percent': (0, 100),
            'wishlist_count': (0, float('inf')),
            'review_count': (0, float('inf')),
            'achievement_count': (0, 10000),
            'trading_cards': (0, 1),
            'metacritic_score': (0, 100)
        }
        
        for column, (min_val, max_val) in validation_rules.items():
            if column in df.columns:
                initial_count = len(df)
                
                invalid_min = df[column] < min_val
                invalid_max = df[column] > max_val
                invalid_rows = invalid_min | invalid_max
                
                if invalid_rows.sum() > 0:
                    self.log_processing_step(f"Found {invalid_rows.sum()} invalid values in {column}")
                    df.loc[invalid_min, column] = min_val
                    df.loc[invalid_max, column] = max_val
                    self.log_processing_step(f"Clamped {column} values to valid range [{min_val}, {max_val}]")
                    
        return df
        
    def clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        text_columns = ['name', 'developer', 'publisher', 'tags', 'description']
        
        for column in text_columns:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = df[column].str.strip()
                df[column] = df[column].replace('', np.nan)
                df[column] = df[column].replace('nan', np.nan)
                
                if column == 'tags':
                    df[column] = df[column].str.replace(r'[^\w\s,]', '', regex=True)
                    df['tag_count'] = df[column].str.count(',') + 1
                    df['tag_count'] = df['tag_count'].fillna(0)
                    
                elif column in ['developer', 'publisher']:
                    df[column] = df[column].str.title()
                    df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
                    
                elif column == 'name':
                    df['name_length'] = df[column].str.len()
                    df['name_word_count'] = df[column].str.split().str.len()
                    df['has_numbers_in_name'] = df[column].str.contains(r'\d', regex=True)
                    df['has_special_chars'] = df[column].str.contains(r'[^\w\s]', regex=True)
                    
                self.log_processing_step(f"Cleaned text field {column}")
                
        return df
        
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_processing_step("Creating derived features")
        
        if 'demo_release_date' in df.columns and 'release_date' in df.columns:
            df['demo_to_release_gap'] = (df['release_date'] - pd.to_datetime(df['demo_release_date'])).dt.days
            df['demo_timing_category'] = pd.cut(df['demo_to_release_gap'], 
                                               bins=[-float('inf'), -30, 0, 30, float('inf')],
                                               labels=['Early_Demo', 'Pre_Release', 'Simultaneous', 'Post_Release'])
                                               
        if 'wishlist_count' in df.columns and 'revenue' in df.columns:
            df['wishlist_to_revenue_ratio'] = df['revenue'] / (df['wishlist_count'] + 1)
            df['high_wishlist'] = df['wishlist_count'] > df['wishlist_count'].quantile(0.75)
            
        if 'price' in df.columns:
            df['price_category'] = pd.cut(df['price'], 
                                        bins=[0, 5, 15, 30, 60, float('inf')],
                                        labels=['Free/Cheap', 'Budget', 'Mid_Range', 'Premium', 'Expensive'])
            df['is_free'] = df['price'] == 0
            
        if 'revenue' in df.columns:
            df['log_revenue'] = np.log1p(df['revenue'])
            df['revenue_success'] = df['revenue'] > df['revenue'].quantile(0.8)
            df['revenue_quartile'] = pd.qcut(df['revenue'], q=4, labels=['Low', 'Medium_Low', 'Medium_High', 'High'])
            
        feature_interactions = [
            ('demo_available', 'price_category'),
            ('demo_available', 'release_season'),
            ('high_wishlist', 'demo_available'),
            ('price_category', 'release_quarter')
        ]
        
        for feat1, feat2 in feature_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f'{feat1}_x_{feat2}'
                df[interaction_name] = df[feat1].astype(str) + '_' + df[feat2].astype(str)
                self.log_processing_step(f"Created interaction feature {interaction_name}")
                
        return df
        
    def classify_demo_timing(self, demo_data: Dict) -> str:
        if not demo_data.get('demo_available') or not demo_data.get('demo_release_date'):
            return 'no_demo'
            
        try:
            demo_date_str = demo_data['demo_release_date']
            game_date_str = demo_data['full_game_release_date']
            
            date_formats = ['%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
            
            demo_date = None
            game_date = None
            
            for fmt in date_formats:
                try:
                    demo_date = datetime.strptime(demo_date_str, fmt)
                    break
                except ValueError:
                    continue
                    
            for fmt in date_formats:
                try:
                    game_date = datetime.strptime(game_date_str, fmt)
                    break
                except ValueError:
                    continue
                    
            if demo_date is None or game_date is None:
                return 'unknown'
                
            time_diff = (game_date - demo_date).days
            
            if time_diff > 7:
                return 'pre_release'
            elif time_diff >= -7:
                return 'simultaneous'
            else:
                return 'post_release'
                
        except Exception as e:
            self.log_processing_step(f"Error classifying demo timing: {str(e)}", "WARNING")
            return 'unknown'

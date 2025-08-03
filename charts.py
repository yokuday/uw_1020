import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings

class SteamVisualizationEngine:
    def __init__(self, style='seaborn-v0_8', palette='husl', figsize=(12, 8)):
        plt.style.use(style)
        sns.set_palette(palette)
        self.default_figsize = figsize
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff871f',
            'info': '#17a2b8',
            'demo': '#e377c2',
            'no_demo': '#7f7f7f',
            'pre_release': '#2ca02c',
            'simultaneous': '#ff7f0e',
            'post_release': '#d62728'
        }
        
        self.plot_config = {
            'font_family': 'Arial',
            'title_size': 16,
            'label_size': 12,
            'tick_size': 10,
            'legend_size': 10,
            'dpi': 300,
            'transparent': False
        }
        
        warnings.filterwarnings('ignore', category=UserWarning)
        
    def create_revenue_comparison_plot(self, df, save_path=None):
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        demo_revenue = df[df['demo_available'] == True]['revenue'].dropna()
        non_demo_revenue = df[df['demo_available'] == False]['revenue'].dropna()
        
        box_data = [non_demo_revenue, demo_revenue]
        bp = ax1.boxplot(box_data, labels=['No Demo', 'With Demo'], patch_artist=True,
                        boxprops=dict(facecolor=self.color_scheme['primary'], alpha=0.7),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
        
        ax1.set_ylabel('Revenue ($)', fontsize=self.plot_config['label_size'])
        ax1.set_title('Revenue Distribution: Demo vs Non-Demo Games', fontsize=self.plot_config['title_size'])
        ax1.tick_params(axis='both', which='major', labelsize=self.plot_config['tick_size'])
        ax1.grid(True, alpha=0.3)
        
        median_no_demo = non_demo_revenue.median()
        median_demo = demo_revenue.median()
        improvement = ((median_demo - median_no_demo) / median_no_demo) * 100
        ax1.text(0.5, 0.95, f'Median Improvement: {improvement:.1f}%', 
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2 = fig.add_subplot(gs[0, 1])
        log_demo = np.log1p(demo_revenue)
        log_non_demo = np.log1p(non_demo_revenue)
        
        ax2.hist(log_non_demo, bins=50, alpha=0.7, label='No Demo', color=self.color_scheme['no_demo'])
        ax2.hist(log_demo, bins=50, alpha=0.7, label='With Demo', color=self.color_scheme['demo'])
        ax2.set_xlabel('Log(Revenue + 1)', fontsize=self.plot_config['label_size'])
        ax2.set_ylabel('Frequency', fontsize=self.plot_config['label_size'])
        ax2.set_title('Revenue Distribution (Log Scale)', fontsize=self.plot_config['title_size'])
        ax2.legend(fontsize=self.plot_config['legend_size'])
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 0])
        percentiles = np.arange(0, 101, 5)
        demo_percentiles = np.percentile(demo_revenue, percentiles)
        non_demo_percentiles = np.percentile(non_demo_revenue, percentiles)
        
        ax3.plot(percentiles, demo_percentiles, label='With Demo', color=self.color_scheme['demo'], linewidth=2)
        ax3.plot(percentiles, non_demo_percentiles, label='No Demo', color=self.color_scheme['no_demo'], linewidth=2)
        ax3.set_xlabel('Percentile', fontsize=self.plot_config['label_size'])
        ax3.set_ylabel('Revenue ($)', fontsize=self.plot_config['label_size'])
        ax3.set_title('Revenue Percentile Comparison', fontsize=self.plot_config['title_size'])
        ax3.legend(fontsize=self.plot_config['legend_size'])
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        violin_data = [non_demo_revenue, demo_revenue]
        parts = ax4.violinplot(violin_data, positions=[1, 2], showmeans=True, showmedians=True)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts.__dict__:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(1)
        
        for i, pc in enumerate(parts['bodies']):
            if i == 0:
                pc.set_facecolor(self.color_scheme['no_demo'])
            else:
                pc.set_facecolor(self.color_scheme['demo'])
            pc.set_alpha(0.7)
            
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['No Demo', 'With Demo'])
        ax4.set_ylabel('Revenue ($)', fontsize=self.plot_config['label_size'])
        ax4.set_title('Revenue Density Distribution', fontsize=self.plot_config['title_size'])
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Revenue Analysis: Demo Impact', fontsize=18, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_config['dpi'], bbox_inches='tight', 
                       transparent=self.plot_config['transparent'])
        
        return fig
        
    def create_correlation_heatmap(self, correlation_matrix, p_values=None, save_path=None):
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        cmap = sns.diverging_palette(250, 10, as_cmap=True, center='light')
        
        heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap, center=0,
                             square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                             fmt='.3f', annot_kws={'fontsize': 8})
        
        if p_values is not None:
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    if not mask[i, j] and i != j:
                        p_val = p_values.iloc[i, j]
                        if p_val < 0.001:
                            significance = '***'
                        elif p_val < 0.01:
                            significance = '**'
                        elif p_val < 0.05:
                            significance = '*'
                        else:
                            significance = ''
                            
                        if significance:
                            ax.text(j + 0.5, i + 0.75, significance, ha='center', va='center',
                                   color='white', fontweight='bold', fontsize=10)
        
        ax.set_title('Correlation Matrix: Game Performance Variables', 
                    fontsize=self.plot_config['title_size'], pad=20)
        
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Pearson Correlation Coefficient', fontsize=self.plot_config['label_size'])
        
        plt.xticks(rotation=45, ha='right', fontsize=self.plot_config['tick_size'])
        plt.yticks(rotation=0, fontsize=self.plot_config['tick_size'])
        
        legend_elements = [
            mpatches.Patch(color='white', label='*** p < 0.001'),
            mpatches.Patch(color='white', label='** p < 0.01'),
            mpatches.Patch(color='white', label='* p < 0.05')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_config['dpi'], bbox_inches='tight')
            
        return fig
        
    def create_demo_timing_analysis(self, df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Demo Timing Strategy Analysis', fontsize=18, y=0.98)
        
        timing_colors = {
            'pre_release': self.color_scheme['pre_release'],
            'simultaneous': self.color_scheme['simultaneous'], 
            'post_release': self.color_scheme['post_release'],
            'no_demo': self.color_scheme['no_demo']
        }
        
        timing_data = df.groupby('demo_timing')['revenue'].apply(list)
        timing_medians = df.groupby('demo_timing')['revenue'].median().sort_values(ascending=False)
        
        ax1 = axes[0, 0]
        box_data = [timing_data[timing] for timing in timing_medians.index]
        bp = ax1.boxplot(box_data, labels=timing_medians.index, patch_artist=True)
        
        for patch, timing in zip(bp['boxes'], timing_medians.index):
            patch.set_facecolor(timing_colors.get(timing, self.color_scheme['primary']))
            patch.set_alpha(0.7)
            
        ax1.set_ylabel('Revenue ($)', fontsize=self.plot_config['label_size'])
        ax1.set_title('Revenue by Demo Timing Strategy', fontsize=self.plot_config['title_size'])
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        conversion_data = df.groupby('demo_timing')['wishlist_count'].median()
        bars = ax2.bar(conversion_data.index, conversion_data.values, 
                      color=[timing_colors.get(timing, self.color_scheme['primary']) for timing in conversion_data.index],
                      alpha=0.8)
        
        for bar, value in zip(bars, conversion_data.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
                    
        ax2.set_ylabel('Median Wishlist Count', fontsize=self.plot_config['label_size'])
        ax2.set_title('Wishlist Engagement by Demo Timing', fontsize=self.plot_config['title_size'])
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = axes[1, 0]
        timing_counts = df['demo_timing'].value_counts()
        colors = [timing_colors.get(timing, self.color_scheme['primary']) for timing in timing_counts.index]
        
        wedges, texts, autotexts = ax3.pie(timing_counts.values, labels=timing_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Distribution of Demo Timing Strategies', fontsize=self.plot_config['title_size'])
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
            
        ax4 = axes[1, 1]
        if 'demo_to_release_gap' in df.columns:
            demo_gap_data = df[df['demo_to_release_gap'].notna()]['demo_to_release_gap']
            ax4.hist(demo_gap_data, bins=30, color=self.color_scheme['info'], alpha=0.7, edgecolor='black')
            ax4.axvline(demo_gap_data.median(), color='red', linestyle='--', linewidth=2, 
                       label=f'Median: {demo_gap_data.median():.0f} days')
            ax4.set_xlabel('Days Between Demo and Full Release', fontsize=self.plot_config['label_size'])
            ax4.set_ylabel('Frequency', fontsize=self.plot_config['label_size'])
            ax4.set_title('Demo Release Gap Distribution', fontsize=self.plot_config['title_size'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Demo Gap Data\nNot Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Demo Release Gap Analysis', fontsize=self.plot_config['title_size'])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_config['dpi'], bbox_inches='tight')
            
        return fig
        
    def create_interactive_scatter_plot(self, df, x_col='wishlist_count', y_col='revenue', 
                                       color_col='demo_available', size_col='price'):
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                        hover_data=['name', 'developer', 'release_date'] if 'name' in df.columns else None,
                        title=f'Interactive Analysis: {y_col.title()} vs {x_col.title()}',
                        labels={x_col: x_col.replace('_', ' ').title(),
                               y_col: y_col.replace('_', ' ').title(),
                               color_col: color_col.replace('_', ' ').title(),
                               size_col: size_col.replace('_', ' ').title()},
                        color_discrete_map={True: self.color_scheme['demo'], 
                                          False: self.color_scheme['no_demo']})
        
        fig.update_layout(
            font_family=self.plot_config['font_family'],
            title_font_size=self.plot_config['title_size'],
            xaxis_title_font_size=self.plot_config['label_size'],
            yaxis_title_font_size=self.plot_config['label_size'],
            legend_title_font_size=self.plot_config['legend_size'],
            width=800,
            height=600
        )
        
        if x_col in ['revenue', 'price', 'wishlist_count']:
            fig.update_xaxes(type='log')
        if y_col in ['revenue', 'price', 'wishlist_count']:
            fig.update_yaxes(type='log')
            
        return fig
        
    def create_genre_analysis_plot(self, df, save_path=None):
        if 'genre' not in df.columns:
            print("Genre data not available for analysis")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre-Specific Demo Performance Analysis', fontsize=18, y=0.98)
        
        genre_demo_stats = df.groupby(['genre', 'demo_available']).agg({
            'revenue': ['count', 'median', 'mean'],
            'wishlist_count': 'median'
        }).reset_index()
        
        genre_demo_stats.columns = ['genre', 'demo_available', 'game_count', 'median_revenue', 
                                   'mean_revenue', 'median_wishlist']
        
        top_genres = df['genre'].value_counts().head(8).index
        genre_subset = genre_demo_stats[genre_demo_stats['genre'].isin(top_genres)]
        
        ax1 = axes[0, 0]
        demo_data = genre_subset[genre_subset['demo_available'] == True]
        no_demo_data = genre_subset[genre_subset['demo_available'] == False]
        
        x = np.arange(len(top_genres))
        width = 0.35
        
        demo_revenues = [demo_data[demo_data['genre'] == genre]['median_revenue'].values[0] 
                        if len(demo_data[demo_data['genre'] == genre]) > 0 else 0 
                        for genre in top_genres]
        no_demo_revenues = [no_demo_data[no_demo_data['genre'] == genre]['median_revenue'].values[0] 
                           if len(no_demo_data[no_demo_data['genre'] == genre]) > 0 else 0 
                           for genre in top_genres]
        
        bars1 = ax1.bar(x - width/2, demo_revenues, width, label='With Demo', 
                       color=self.color_scheme['demo'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, no_demo_revenues, width, label='No Demo', 
                       color=self.color_scheme['no_demo'], alpha=0.8)
        
        ax1.set_xlabel('Genre', fontsize=self.plot_config['label_size'])
        ax1.set_ylabel('Median Revenue ($)', fontsize=self.plot_config['label_size'])
        ax1.set_title('Revenue Performance by Genre', fontsize=self.plot_config['title_size'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_genres, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        return fig

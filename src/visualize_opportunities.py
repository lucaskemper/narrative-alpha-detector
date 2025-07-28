import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set up modern styling
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

class OpportunityVisualizer:
    def __init__(self, csv_path='results/scored_markets.csv'):
        """Initialize the visualizer with the scored markets data."""
        self.df = pd.read_csv(csv_path)
        self.df['end_date'] = pd.to_datetime(self.df['end_date'])
        # Handle timezone-aware datetime objects
        now = pd.Timestamp.now().tz_localize(None)
        self.df['days_to_expiry'] = (self.df['end_date'].dt.tz_localize(None) - now).dt.days
        
        # Modern color palette
        self.colors = {
            'buy_yes': '#2E8B57',  # Sea Green
            'buy_no': '#DC143C',   # Crimson
            'background': '#f8f9fa',
            'text': '#2c3e50',
            'grid': '#e9ecef',
            'accent': '#3498db'
        }
        
    def get_top_opportunities(self, n=10, min_abs_mispricing=0.05):
        """Get top opportunities based on absolute mispricing."""
        filtered_df = self.df[self.df['abs_mispricing'] >= min_abs_mispricing].copy()
        return filtered_df.nlargest(n, 'abs_mispricing')
    
    def plot_top_opportunities_bar(self, n=10, save_path='results/top_opportunities_bar.png'):
        """Create a beautiful bar chart of top opportunities by mispricing."""
        top_opps = self.get_top_opportunities(n)
        
        # Create figure with modern styling
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create horizontal bar chart with gradient colors
        y_pos = np.arange(len(top_opps))
        colors = [self.colors['buy_yes'] if x > 0 else self.colors['buy_no'] for x in top_opps['mispricing']]
        
        bars = ax.barh(y_pos, top_opps['abs_mispricing'], 
                       color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels on bars with better positioning
        for i, (bar, mispricing) in enumerate(zip(bars, top_opps['abs_mispricing'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{mispricing:.3f}', va='center', fontweight='bold', 
                   fontsize=11, color=self.colors['text'])
        
        # Customize the plot with modern styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels([title[:45] + '...' if len(title) > 45 else title 
                           for title in top_opps['title']], fontsize=10)
        ax.set_xlabel('Absolute Mispricing', fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.set_title(f'🎯 Top {n} Trading Opportunities by Mispricing', 
                    fontsize=18, fontweight='bold', color=self.colors['text'], pad=20)
        
        # Add direction indicators with better styling
        for i, direction in enumerate(top_opps['direction']):
            direction_text = 'BUY YES' if direction == 'buy_yes' else 'BUY NO'
            color = self.colors['buy_yes'] if direction == 'buy_yes' else self.colors['buy_no']
            ax.text(-0.05, i, direction_text, ha='right', va='center', 
                   fontweight='bold', fontsize=10, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
        
        # Add grid and styling
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor(self.colors['background'])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['buy_yes'], label='Buy Yes', alpha=0.8),
            Patch(facecolor=self.colors['buy_no'], label='Buy No', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return top_opps
    
    def plot_mispricing_vs_prior(self, save_path='results/mispricing_vs_prior.png'):
        """Create a beautiful scatter plot of mispricing vs prior probability."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Color points by direction with better styling
        colors = [self.colors['buy_yes'] if d == 'buy_yes' else self.colors['buy_no'] for d in self.df['direction']]
        sizes = self.df['abs_mispricing'] * 2000  # Scale for visibility
        
        scatter = ax.scatter(self.df['px_prior'], self.df['mispricing'], 
                           c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=1.5)
        
        # Add labels for top opportunities with better styling
        top_opps = self.get_top_opportunities(5)
        for _, row in top_opps.iterrows():
            color = self.colors['buy_yes'] if row['direction'] == 'buy_yes' else self.colors['buy_no']
            ax.annotate(row['title'][:25] + '...', 
                       (row['px_prior'], row['mispricing']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                                edgecolor=color, linewidth=1))
        
        ax.set_xlabel('Prior Probability', fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.set_ylabel('Mispricing', fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.set_title('📊 Mispricing vs Prior Probability Analysis', fontsize=18, fontweight='bold', 
                    color=self.colors['text'], pad=20)
        
        # Add grid with better styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['background'])
        
        # Add legend with better styling
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['buy_yes'], label='Buy Yes', alpha=0.7),
            Patch(facecolor=self.colors['buy_no'], label='Buy No', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Add trend line
        z = np.polyfit(self.df['px_prior'], self.df['mispricing'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['px_prior'], p(self.df['px_prior']), "k--", alpha=0.5, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_opportunity_heatmap(self, save_path='results/opportunity_heatmap.png'):
        """Create a beautiful heatmap of opportunities by various metrics."""
        top_opps = self.get_top_opportunities(15)
        
        # Prepare data for heatmap
        metrics_df = top_opps[['abs_mispricing', 'px_prior', 'prob_yes', 'days_to_expiry']].copy()
        metrics_df.columns = ['Mispricing', 'Prior', 'Market_Prob', 'Days_Left']
        
        # Normalize the data for better visualization
        metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
        
        # Create figure with modern styling
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create heatmap with better styling
        sns.heatmap(metrics_normalized.T, 
                   xticklabels=[title[:25] + '...' for title in top_opps['title']],
                   yticklabels=metrics_normalized.columns,
                   annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Normalized Value'}, ax=ax)
        
        ax.set_title('🔥 Opportunity Metrics Heatmap', fontsize=18, fontweight='bold', 
                    color=self.colors['text'], pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_interactive_dashboard(self, save_path='results/opportunity_dashboard.html'):
        """Create a beautiful interactive Plotly dashboard."""
        top_opps = self.get_top_opportunities(15)
        
        # Create subplots with better layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📈 Top Opportunities by Mispricing', 
                          '📊 Mispricing vs Prior Probability',
                          '⏰ Days to Expiry vs Mispricing',
                          '📋 Market Probability Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Bar chart of top opportunities with better styling
        colors = [self.colors['buy_yes'] if x > 0 else self.colors['buy_no'] for x in top_opps['mispricing']]
        fig.add_trace(
            go.Bar(x=top_opps['title'], y=top_opps['abs_mispricing'],
                   marker_color=colors, name='Mispricing',
                   hovertemplate='<b>%{x}</b><br>Mispricing: %{y:.3f}<extra></extra>'),
            row=1, col=1
        )
        
        # Scatter plot: mispricing vs prior with better styling
        fig.add_trace(
            go.Scatter(x=top_opps['px_prior'], y=top_opps['mispricing'],
                      mode='markers', 
                      marker=dict(size=top_opps['abs_mispricing']*100, 
                                color=[self.colors['buy_yes'] if x > 0 else self.colors['buy_no'] for x in top_opps['mispricing']],
                                line=dict(color='white', width=2)),
                      text=top_opps['title'], name='Mispricing vs Prior',
                      hovertemplate='<b>%{text}</b><br>Prior: %{x:.3f}<br>Mispricing: %{y:.3f}<extra></extra>'),
            row=1, col=2
        )
        
        # Scatter plot: days to expiry vs mispricing
        fig.add_trace(
            go.Scatter(x=top_opps['days_to_expiry'], y=top_opps['abs_mispricing'],
                      mode='markers', 
                      marker=dict(size=top_opps['px_prior']*100,
                                color=[self.colors['buy_yes'] if x > 0 else self.colors['buy_no'] for x in top_opps['mispricing']],
                                line=dict(color='white', width=2)),
                      text=top_opps['title'], name='Days vs Mispricing',
                      hovertemplate='<b>%{text}</b><br>Days: %{x}<br>Mispricing: %{y:.3f}<extra></extra>'),
            row=2, col=1
        )
        
        # Histogram of market probabilities
        fig.add_trace(
            go.Histogram(x=top_opps['prob_yes'], nbinsx=10, name='Market Prob Distribution',
                        marker_color=self.colors['accent'],
                        hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout with modern styling
        fig.update_layout(
            height=900,
            title_text="🚀 Trading Opportunities Dashboard",
            title_font_size=24,
            title_font_color=self.colors['text'],
            showlegend=True,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor='white',
            font=dict(color=self.colors['text'])
        )
        
        # Update axes styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        
        fig.write_html(save_path)
        
        return fig
    
    def generate_summary_report(self, save_path='results/opportunity_summary.txt'):
        """Generate a beautiful text summary of top opportunities."""
        top_opps = self.get_top_opportunities(10)
        
        with open(save_path, 'w') as f:
            f.write("🎯 TOP TRADING OPPORTUNITIES SUMMARY 🎯\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 Total markets analyzed: {len(self.df)}\n")
            f.write(f"💡 Markets with significant mispricing (>=0.05): {len(self.df[self.df['abs_mispricing'] >= 0.05])}\n\n")
            
            f.write("🏆 TOP 10 OPPORTUNITIES:\n")
            f.write("=" * 60 + "\n")
            
            for i, (_, row) in enumerate(top_opps.iterrows(), 1):
                direction_emoji = "✅" if row['direction'] == 'buy_yes' else "❌"
                f.write(f"\n{i}. {direction_emoji} {row['title']}\n")
                f.write(f"   🔗 URL: {row['url']}\n")
                f.write(f"   📈 Direction: {row['direction'].upper()}\n")
                f.write(f"   🎯 Market Probability: {row['prob_yes']:.3f}\n")
                f.write(f"   📊 Prior Probability: {row['px_prior']:.3f}\n")
                f.write(f"   💰 Mispricing: {row['mispricing']:.3f}\n")
                f.write(f"   ⏰ Days to Expiry: {row['days_to_expiry']}\n")
                f.write(f"   📈 Expected Value: {row['mispricing']:.3f}\n")
                f.write("-" * 50 + "\n")
            
            # Add statistics with emojis
            f.write(f"\n📈 STATISTICS:\n")
            f.write(f"📊 Average mispricing: {self.df['abs_mispricing'].mean():.3f}\n")
            f.write(f"📊 Median mispricing: {self.df['abs_mispricing'].median():.3f}\n")
            f.write(f"📊 Max mispricing: {self.df['abs_mispricing'].max():.3f}\n")
            f.write(f"✅ Markets favoring YES: {len(self.df[self.df['direction'] == 'buy_yes'])}\n")
            f.write(f"❌ Markets favoring NO: {len(self.df[self.df['direction'] == 'buy_no'])}\n")
    
    def run_full_analysis(self):
        """Run the complete analysis and generate all visualizations."""
        print("🔍 Analyzing trading opportunities...")
        print("🎨 Creating beautiful visualizations...")
        
        # Generate all visualizations
        top_opps = self.plot_top_opportunities_bar()
        self.plot_mispricing_vs_prior()
        self.plot_opportunity_heatmap()
        self.create_interactive_dashboard()
        self.generate_summary_report()
        
        print("✅ Analysis complete! Generated files:")
        print("  📊 results/top_opportunities_bar.png")
        print("  📈 results/mispricing_vs_prior.png")
        print("  🔥 results/opportunity_heatmap.png")
        print("  🚀 results/opportunity_dashboard.html")
        print("  📋 results/opportunity_summary.txt")
        
        return top_opps

if __name__ == "__main__":
    # Create visualizer and run analysis
    visualizer = OpportunityVisualizer()
    top_opportunities = visualizer.run_full_analysis()
    
    print(f"\n🎯 Top 5 Opportunities:")
    for i, (_, row) in enumerate(top_opportunities.head().iterrows(), 1):
        direction_emoji = "✅" if row['direction'] == 'buy_yes' else "❌"
        print(f"{i}. {direction_emoji} {row['title'][:50]}...")
        print(f"   📈 Direction: {row['direction'].upper()}")
        print(f"   💰 Mispricing: {row['abs_mispricing']:.3f}")
        print() 
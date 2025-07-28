#!/usr/bin/env python3
"""
Beautiful command-line interface for analyzing trading opportunities.
"""

import argparse
import sys
from visualize_opportunities import OpportunityVisualizer

def print_banner():
    """Print a beautiful banner."""
    print("🎯" + "="*60 + "🎯")
    print("🚀 TRADING OPPORTUNITIES ANALYZER 🚀")
    print("🎯" + "="*60 + "🎯")
    print()

def main():
    parser = argparse.ArgumentParser(description='🎯 Analyze trading opportunities from scored markets data')
    parser.add_argument('--csv', default='results/scored_markets.csv', 
                       help='📁 Path to scored markets CSV file')
    parser.add_argument('--top', type=int, default=10, 
                       help='🏆 Number of top opportunities to show')
    parser.add_argument('--min-mispricing', type=float, default=0.05,
                       help='💰 Minimum absolute mispricing threshold')
    parser.add_argument('--output-dir', default='results',
                       help='📂 Output directory for generated files')
    parser.add_argument('--interactive', action='store_true',
                       help='🌐 Open interactive dashboard in browser')
    parser.add_argument('--quick', action='store_true',
                       help='⚡ Quick analysis without visualizations')
    
    args = parser.parse_args()
    
    try:
        print_banner()
        
        # Initialize visualizer
        print("🔍 Loading data...")
        visualizer = OpportunityVisualizer(args.csv)
        
        # Get top opportunities
        top_opps = visualizer.get_top_opportunities(args.top, args.min_mispricing)
        
        print(f"\n🎯 TOP {len(top_opps)} TRADING OPPORTUNITIES")
        print("=" * 60)
        
        for i, (_, row) in enumerate(top_opps.iterrows(), 1):
            direction_emoji = "✅" if row['direction'] == 'buy_yes' else "❌"
            print(f"\n{i}. {direction_emoji} {row['title']}")
            print(f"   📈 Direction: {row['direction'].upper()}")
            print(f"   🎯 Market Probability: {row['prob_yes']:.3f}")
            print(f"   📊 Prior Probability: {row['px_prior']:.3f}")
            print(f"   💰 Mispricing: {row['mispricing']:.3f}")
            print(f"   ⏰ Days to Expiry: {row['days_to_expiry']}")
            print(f"   🔗 URL: {row['url']}")
            print("-" * 50)
        
        if not args.quick:
            # Generate visualizations
            print("\n🎨 Generating beautiful visualizations...")
            visualizer.plot_top_opportunities_bar(args.top, f'{args.output_dir}/top_opportunities_bar.png')
            visualizer.plot_mispricing_vs_prior(f'{args.output_dir}/mispricing_vs_prior.png')
            visualizer.plot_opportunity_heatmap(f'{args.output_dir}/opportunity_heatmap.png')
            visualizer.create_interactive_dashboard(f'{args.output_dir}/opportunity_dashboard.html')
            visualizer.generate_summary_report(f'{args.output_dir}/opportunity_summary.txt')
            
            print("✅ Analysis complete!")
            print(f"📁 Files saved to: {args.output_dir}/")
            
            if args.interactive:
                import webbrowser
                import os
                dashboard_path = os.path.abspath(f'{args.output_dir}/opportunity_dashboard.html')
                print(f"🌐 Opening interactive dashboard: {dashboard_path}")
                webbrowser.open(f'file://{dashboard_path}')
        else:
            print("\n⚡ Quick analysis complete!")
        
        # Print summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   📈 Average mispricing: {visualizer.df['abs_mispricing'].mean():.3f}")
        print(f"   📈 Median mispricing: {visualizer.df['abs_mispricing'].median():.3f}")
        print(f"   📈 Max mispricing: {visualizer.df['abs_mispricing'].max():.3f}")
        print(f"   ✅ Buy YES opportunities: {len(visualizer.df[visualizer.df['direction'] == 'buy_yes'])}")
        print(f"   ❌ Buy NO opportunities: {len(visualizer.df[visualizer.df['direction'] == 'buy_no'])}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
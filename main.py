#!/usr/bin/env python3
"""
🎯 Narrative Alpha Detector - Main Pipeline
==========================================

This script orchestrates the complete analysis pipeline:
1. Scrape Polymarket data
2. Generate priors using AI
3. Score markets for mispricing
4. Visualize top opportunities

Usage:
    python main.py                    # Run full pipeline
    python main.py --visualize-only   # Only create visualizations
    python main.py --scrape-only      # Only scrape new data
    python main.py --quick            # Quick analysis without visualizations
"""

import argparse
import sys
import os
from datetime import datetime
import subprocess

def print_banner():
    """Print a beautiful banner."""
    print("🎯" + "="*70 + "🎯")
    print("🚀 NARRATIVE ALPHA DETECTOR - MAIN PIPELINE 🚀")
    print("🎯" + "="*70 + "🎯")
    print()

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    try:
        import pandas
        import matplotlib
        import seaborn
        import plotly
        import requests
        import openai
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def run_scraping():
    """Run the Polymarket data scraper."""
    print("\n📊 Step 1: Scraping Polymarket data...")
    try:
        result = subprocess.run([sys.executable, "src/polymarket_api_scraper.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Data scraping completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data scraping failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_prior_generation():
    """Run the prior generation using AI."""
    print("\n🤖 Step 2: Generating priors using AI...")
    try:
        result = subprocess.run([sys.executable, "src/generate_priors.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Prior generation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Prior generation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_mispricing_scoring():
    """Run the mispricing scoring analysis."""
    print("\n📈 Step 3: Scoring markets for mispricing...")
    try:
        result = subprocess.run([sys.executable, "src/score_mispricing.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Mispricing scoring completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Mispricing scoring failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_visualization():
    """Run the opportunity visualization."""
    print("\n🎨 Step 4: Creating beautiful visualizations...")
    try:
        result = subprocess.run([sys.executable, "src/visualize_opportunities.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Visualizations created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Visualization failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_quick_analysis():
    """Run a quick analysis without visualizations."""
    print("\n⚡ Running quick analysis...")
    try:
        result = subprocess.run([sys.executable, "src/analyze_opportunities.py", "--quick"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quick analysis failed: {e}")
        return False

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "results/scored_markets.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing data files: {missing_files}")
        return False
    
    print("✅ All required data files found!")
    return True

def print_summary():
    """Print a summary of generated files."""
    print("\n📁 GENERATED FILES SUMMARY:")
    print("=" * 50)
    
    files_to_check = [
        ("📊 Raw Data", "data/polymarket_raw.json"),
        ("📈 Scored Markets", "results/scored_markets.csv"),
        ("📊 Top Opportunities Chart", "results/top_opportunities_bar.png"),
        ("📈 Mispricing Analysis", "results/mispricing_vs_prior.png"),
        ("🔥 Metrics Heatmap", "results/opportunity_heatmap.png"),
        ("🚀 Interactive Dashboard", "results/opportunity_dashboard.html"),
        ("📋 Summary Report", "results/opportunity_summary.txt")
    ]
    
    for description, file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f}MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"✅ {description}: {file_path} ({size_str})")
        else:
            print(f"❌ {description}: {file_path} (missing)")

def main():
    parser = argparse.ArgumentParser(description='🎯 Narrative Alpha Detector - Main Pipeline')
    parser.add_argument('--visualize-only', action='store_true',
                       help='🎨 Only create visualizations (skip scraping and analysis)')
    parser.add_argument('--scrape-only', action='store_true',
                       help='📊 Only scrape new data (skip analysis and visualization)')
    parser.add_argument('--quick', action='store_true',
                       help='⚡ Quick analysis without visualizations')
    parser.add_argument('--check-deps', action='store_true',
                       help='🔍 Check dependencies only')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ Dependencies check completed!")
        return
    
    start_time = datetime.now()
    
    try:
        if args.scrape_only:
            # Only scrape data
            if not run_scraping():
                sys.exit(1)
            print("✅ Data scraping completed!")
            
        elif args.visualize_only:
            # Only create visualizations
            if not check_data_files():
                print("❌ Required data files not found. Run full pipeline first.")
                sys.exit(1)
            if not run_visualization():
                sys.exit(1)
            print("✅ Visualizations created!")
            
        elif args.quick:
            # Quick analysis
            if not check_data_files():
                print("❌ Required data files not found. Run full pipeline first.")
                sys.exit(1)
            if not run_quick_analysis():
                sys.exit(1)
            print("✅ Quick analysis completed!")
            
        else:
            # Full pipeline
            print("🚀 Running full analysis pipeline...")
            
            # Step 1: Scrape data
            if not run_scraping():
                sys.exit(1)
            
            # Step 2: Generate priors
            if not run_prior_generation():
                sys.exit(1)
            
            # Step 3: Score mispricing
            if not run_mispricing_scoring():
                sys.exit(1)
            
            # Step 4: Create visualizations
            if not run_visualization():
                sys.exit(1)
            
            print("🎉 Full pipeline completed successfully!")
        
        # Print summary
        print_summary()
        
        # Print timing
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n⏱️  Total execution time: {duration}")
        
        print("\n🎯 Next steps:")
        print("   📊 View the generated visualizations in the results/ folder")
        print("   🌐 Open results/opportunity_dashboard.html for interactive analysis")
        print("   📋 Check results/opportunity_summary.txt for detailed report")
        print("   🚀 Run: python src/analyze_opportunities.py --interactive")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

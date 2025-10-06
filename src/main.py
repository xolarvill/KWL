#!/usr/bin/env python3
"""
Main application to run the Estimation Log Visualizer.

This tool provides both a web interface and a command-line interface
for analyzing and visualizing estimation logs.
"""

import argparse
import sys
import os
from pathlib import Path
from src.log_visualization.web_visualizer import WebLogVisualizer
from src.log_visualization.cli_visualizer import main as cli_main


def main():
    """Main entry point for the log visualizer application."""
    parser = argparse.ArgumentParser(
        description='Estimation Log Visualizer - Analyze and visualize estimation logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run web interface
  python -m src.main --web
  
  # Run web interface on specific port
  python -m src.main --web --port 8080
  
  # Run CLI analysis
  python -m src.main /path/to/logfile.log --summary
        """
    )
    
    # Add arguments for web vs CLI
    parser.add_argument('--web', action='store_true', help='Run web interface')
    parser.add_argument('--port', type=int, default=8080, help='Port for web interface (default: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for web interface (default: 127.0.0.1)')
    
    # Arguments for CLI mode (for backward compatibility)
    parser.add_argument('log_file', nargs='?', help='Path to the log file to analyze (CLI mode)')
    parser.add_argument('log_file2', nargs='?', help='Path to the second log file for comparison (CLI mode)')
    parser.add_argument('--filter-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help='Minimum log level to display')
    parser.add_argument('--search', type=str, help='Search term to filter log entries')
    parser.add_argument('--iteration', type=int, nargs='*', help='Filter by specific iteration numbers')
    parser.add_argument('--show-cache', action='store_true', default=True, help='Show cache operations')
    parser.add_argument('--hide-cache', action='store_false', dest='show_cache', help='Hide cache operations')
    parser.add_argument('--show-bellman', action='store_true', default=True, help='Show Bellman operations')
    parser.add_argument('--hide-bellman', action='store_false', dest='show_bellman', help='Hide Bellman operations')
    parser.add_argument('--summary', action='store_true', help='Show summary report')
    parser.add_argument('--performance', action='store_true', help='Show performance analysis')
    parser.add_argument('--timeline', action='store_true', help='Show timeline view')
    parser.add_argument('--compare', action='store_true', help='Compare two log files')
    parser.add_argument('--max-entries', type=int, default=50, help='Maximum number of entries to show in timeline')
    parser.add_argument('--export', type=str, help='Export filtered logs to specified file')
    
    args = parser.parse_args()
    
    # Determine mode based on arguments
    if args.web:
        # Run web interface
        print(f"Starting Estimation Log Visualizer web interface...")
        print(f"Access the interface at: http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server.")
        
        # Create and run web visualizer
        web_viz = WebLogVisualizer(log_dir="progress/log")
        web_viz.run(host=args.host, port=args.port, debug=False)
    
    elif args.log_file or any([
        args.search, args.iteration, args.summary, args.performance, 
        args.timeline, args.export, args.compare
    ]):
        # Run CLI mode
        cli_main()
    
    else:
        # No arguments provided, show help
        parser.print_help()
        print("\n" + "="*60)
        print("Welcome to the Estimation Log Visualizer!")
        print("="*60)
        print("You can use this tool in two ways:")
        print("")
        print("1. Web Interface (recommended):")
        print("   python -m src.main --web [--port PORT]")
        print("   This opens a browser-based interface for log analysis")
        print("")
        print("2. Command Line Interface:")
        print("   python -m src.main [LOG_FILE] [OPTIONS]")
        print("   This provides quick analysis directly in the terminal")
        print("")


if __name__ == "__main__":
    main()
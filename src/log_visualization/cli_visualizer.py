import argparse
import sys
from pathlib import Path
from .log_parser import EstimationLogParser, LogEntry, EMIteration
from .log_visualizer import LogVisualizer, FilterOptions


def create_cli():
    """Create command-line interface for log analysis."""
    parser = argparse.ArgumentParser(description='Estimation Log Analysis Tool')
    parser.add_argument('log_file', nargs='?', help='Path to the log file to analyze')
    
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
    parser.add_argument('--max-entries', type=int, default=50, help='Maximum number of entries to show in timeline')
    parser.add_argument('--export', type=str, help='Export filtered logs to specified file')
    
    return parser


def main():
    """Main CLI function."""
    parser = create_cli()
    args = parser.parse_args()
    
    # If no log file provided, show help
    if not args.log_file:
        parser.print_help()
        print("\nExample usage:")
        print("  python -m src.cli_visualizer /path/to/logfile.log --summary")
        print("  python -m src.cli_visualizer /path/to/logfile.log --timeline --max-entries 100")
        print("  python -m src.cli_visualizer /path/to/logfile.log --search 'Cache HIT' --filter-level WARNING")
        return
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file '{log_path}' does not exist.", file=sys.stderr)
        return
    
    # Initialize parser and visualizer
    parser_instance = EstimationLogParser()
    visualizer = LogVisualizer(parser_instance)
    
    try:
        # Parse the log file
        print(f"Loading log file: {log_path.name}")
        entries = parser_instance.parse_log_file(str(log_path))
        iterations = parser_instance.group_by_iterations(entries)
        
        # Create filter options
        filters = FilterOptions(
            min_level=args.filter_level,
            iteration_filter=args.iteration,
            search_text=args.search or "",
            show_cache_operations=args.show_cache,
            show_bellman_operations=args.show_bellman
        )
        
        # Apply filters if needed
        if any([args.filter_level != 'INFO', args.search, args.iteration, 
                not args.show_cache, not args.show_bellman]):
            filtered_entries = visualizer.filter_entries(entries, filters)
        else:
            filtered_entries = entries
        
        # Determine what to show based on arguments
        show_anything = any([args.summary, args.performance, args.timeline, args.export])
        
        if args.summary or not show_anything:  # Show summary by default if no specific option is given
            summary = visualizer.generate_summary_report(entries, iterations)
            print(summary)
            
            if show_anything:  # Add a separator if showing multiple outputs
                print("\n" + "="*80 + "\n")
        
        if args.performance:
            performance = visualizer.print_performance_analysis(iterations)
            print(performance)
            
            if args.timeline or args.export:
                print("\n" + "="*80 + "\n")
        
        if args.timeline:
            timeline = visualizer.print_timeline_view(filtered_entries, args.max_entries)
            print(timeline)
            
            if args.export:
                print("\n" + "="*80 + "\n")
        
        if args.export:
            visualizer.export_filtered_logs(filtered_entries, args.export)
    
    except Exception as e:
        print(f"Error processing log file: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
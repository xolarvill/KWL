import json
from typing import List, Dict, Any
from dataclasses import dataclass
from .log_parser import EstimationLogParser, LogEntry, EMIteration


@dataclass
class FilterOptions:
    """Options for filtering log entries."""
    min_level: str = "INFO"
    iteration_filter: List[int] = None
    step_filter: List[str] = None
    search_text: str = ""
    show_cache_operations: bool = True
    show_bellman_operations: bool = True
    time_range: tuple = None  # (start_time, end_time)


class LogVisualizer:
    """
    Visualizes estimation logs with filtering and analysis capabilities.
    """
    
    def __init__(self, log_parser: EstimationLogParser):
        self.parser = log_parser
        self.level_weights = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
    
    def filter_entries(self, entries: List[LogEntry], filters: FilterOptions) -> List[LogEntry]:
        """
        Filter log entries based on provided options.
        """
        filtered_entries = []
        
        for entry in entries:
            # Filter by log level
            if self.level_weights[entry.level] < self.level_weights[filters.min_level]:
                continue
            
            # Filter by iteration
            if filters.iteration_filter and entry.iteration not in filters.iteration_filter:
                continue
            
            # Filter by step
            if filters.step_filter and entry.step not in filters.step_filter:
                continue
            
            # Filter by search text
            if filters.search_text and filters.search_text.lower() not in entry.message.lower():
                continue
            
            # Filter cache operations
            if not filters.show_cache_operations and '[Likelihood] Cache' in entry.message:
                continue
            
            # Filter Bellman operations
            if not filters.show_bellman_operations and 'Bellman equation converged' in entry.message:
                continue
            
            # Filter by time range
            if filters.time_range:
                start_time, end_time = filters.time_range
                if entry.timestamp < start_time or entry.timestamp > end_time:
                    continue
            
            filtered_entries.append(entry)
        
        return filtered_entries
    
    def generate_summary_report(self, entries: List[LogEntry], iterations: List[EMIteration]) -> str:
        """
        Generate a summary report of the log analysis.
        """
        stats = self.parser.extract_statistics(entries)
        
        report = []
        report.append("=" * 60)
        report.append("ESTIMATION LOG ANALYSIS REPORT")
        report.append("=" * 60)
        
        if stats['start_time'] and stats['end_time']:
            report.append(f"Analysis Period: {stats['start_time']} to {stats['end_time']}")
            report.append(f"Total Duration: {stats['total_duration']}")
        
        report.append(f"Total Log Entries: {stats['total_entries']}")
        report.append(f"Iterations Completed: {stats['iterations_completed']}")
        report.append(f"Errors: {stats['error_count']}")
        report.append(f"Warnings: {stats['warning_count']}")
        report.append(f"Cache Hits: {stats['cache_stats']['hits']}")
        report.append(f"Cache Misses: {stats['cache_stats']['misses']}")
        report.append("")
        
        # Log likelihood progression
        if stats['log_likelihoods']:
            report.append("LOG LIKELIHOOD PROGRESSION:")
            for ll in stats['log_likelihoods']:
                report.append(f"  Iteration {ll['iteration']}: {ll['value']}")
            report.append("")
        
        # Type probabilities
        if stats['type_probabilities']:
            report.append("TYPE PROBABILITIES:")
            for tp in stats['type_probabilities']:
                report.append(f"  Iteration {tp['iteration']}: {tp['values']}")
            report.append("")
        
        # Iteration breakdown
        report.append("ITERATION BREAKDOWN:")
        for iteration in iterations:
            duration = iteration.end_time - iteration.start_time
            report.append(f"  Iteration {iteration.number}: {duration} duration")
            
            # Count different types of operations
            e_steps = len([e for e in iteration.log_entries if e.step == 'E-step'])
            m_steps = len([e for e in iteration.log_entries if e.step == 'M-step'])
            bellman_ops = len([e for e in iteration.log_entries if 'Bellman' in e.message])
            cache_ops = len([e for e in iteration.log_entries if '[Likelihood] Cache' in e.message])
            
            report.append(f"    - E-step operations: {e_steps}")
            report.append(f"    - M-step operations: {m_steps}")
            report.append(f"    - Bellman operations: {bellman_ops}")
            report.append(f"    - Cache operations: {cache_ops}")
            
            if iteration.log_likelihood is not None:
                report.append(f"    - Log Likelihood: {iteration.log_likelihood}")
            if iteration.change_in_likelihood is not None:
                report.append(f"    - Change in Likelihood: {iteration.change_in_likelihood}")
            if iteration.type_probabilities is not None:
                report.append(f"    - Type Probabilities: {iteration.type_probabilities}")
            
            report.append("")
        
        return "\n".join(report)
    
    def format_entry_for_display(self, entry: LogEntry, include_timestamp: bool = True) -> str:
        """
        Format a log entry for display with color coding.
        """
        timestamp = f"[{entry.timestamp.strftime('%H:%M:%S.%f')[:-3]}] " if include_timestamp else ""
        level = f"{entry.level:<7}"
        iteration = f"(Iter {entry.iteration}) " if entry.iteration else ""
        step = f"[{entry.step}] " if entry.step else ""
        
        formatted = f"{timestamp}{level} {iteration}{step}{entry.message}"
        
        # Color coding for terminal (if supported)
        if entry.level == 'ERROR':
            formatted = f"\033[91m{formatted}\033[0m"  # Red
        elif entry.level == 'WARNING':
            formatted = f"\033[93m{formatted}\033[0m"  # Yellow
        elif 'Cache HIT' in entry.message:
            formatted = f"\033[92m{formatted}\033[0m"  # Green
        elif 'Cache MISS' in entry.message:
            formatted = f"\033[96m{formatted}\033[0m"  # Cyan
        
        return formatted
    
    def print_timeline_view(self, entries: List[LogEntry], max_entries: int = 50) -> str:
        """
        Print a timeline view of the log entries.
        """
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("LOG TIMELINE VIEW")
        output_lines.append("=" * 80)
        
        entries_to_show = entries[:max_entries] if len(entries) > max_entries else entries
        
        for entry in entries_to_show:
            formatted_entry = self.format_entry_for_display(entry)
            output_lines.append(formatted_entry)
        
        if len(entries) > max_entries:
            output_lines.append(f"\n... and {len(entries) - max_entries} more entries")
        
        return "\n".join(output_lines)
    
    def print_performance_analysis(self, iterations: List[EMIteration]) -> str:
        """
        Print performance analysis of the estimation process.
        """
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("PERFORMANCE ANALYSIS")
        output_lines.append("=" * 80)
        
        if not iterations:
            output_lines.append("No iterations found in log.")
            return "\n".join(output_lines)
        
        # Calculate timing statistics
        total_duration = iterations[-1].end_time - iterations[0].start_time
        iteration_durations = [iter.end_time - iter.start_time for iter in iterations]
        avg_duration = sum(iteration_durations, type(iteration_durations[0])()) / len(iteration_durations)
        
        output_lines.append(f"Total Estimation Time: {total_duration}")
        output_lines.append(f"Number of Iterations: {len(iterations)}")
        output_lines.append(f"Average Iteration Time: {avg_duration}")
        output_lines.append("")
        
        # Breakdown by iteration
        output_lines.append("ITERATION TIMING BREAKDOWN:")
        for i, iteration in enumerate(iterations):
            duration = iteration.end_time - iteration.start_time
            output_lines.append(f"  Iteration {iteration.number}: {duration} ({duration / avg_duration:.2f}x avg)")
            
            # Count different operations
            cache_hits = len([e for e in iteration.log_entries if 'Cache HIT' in e.message])
            cache_misses = len([e for e in iteration.log_entries if 'Cache MISS' in e.message])
            bellman_ops = len([e for e in iteration.log_entries if 'Bellman' in e.message])
            
            if cache_hits + cache_misses > 0:
                hit_rate = cache_hits / (cache_hits + cache_misses) * 100
                output_lines.append(f"    - Cache Hit Rate: {hit_rate:.2f}% ({cache_hits} hits, {cache_misses} misses)")
            
            output_lines.append(f"    - Bellman Operations: {bellman_ops}")
            
            if iteration.log_likelihood is not None:
                output_lines.append(f"    - Log Likelihood: {iteration.log_likelihood}")
        
        return "\n".join(output_lines)
    
    def export_filtered_logs(self, entries: List[LogEntry], output_path: str) -> None:
        """
        Export filtered log entries to a file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry.timestamp} - {entry.level} - ")
                if entry.iteration:
                    f.write(f"[Iter {entry.iteration}] ")
                if entry.step:
                    f.write(f"[{entry.step}] ")
                f.write(f"{entry.message}\n")
        
        print(f"Filtered logs exported to {output_path}")
    
    def generate_comparison_report(self, comparison_result: Dict[str, Any]) -> str:
        """
        Generate a comparison report of two log files.
        """
        report = []
        report.append("=" * 80)
        report.append("LOG COMPARISON REPORT")
        report.append("=" * 80)
        
        # Basic statistics comparison
        log1_stats = comparison_result.get('log1_stats', {})
        log2_stats = comparison_result.get('log2_stats', {})
        
        report.append("BASIC STATISTICS COMPARISON:")
        report.append(f"  Log 1 Total Entries: {log1_stats.get('total_entries', 0)}")
        report.append(f"  Log 2 Total Entries: {log2_stats.get('total_entries', 0)}")
        
        duration1 = log1_stats.get('total_duration')
        duration2 = log2_stats.get('total_duration')
        report.append(f"  Log 1 Duration: {duration1}")
        report.append(f"  Log 2 Duration: {duration2}")
        
        report.append(f"  Log 1 Errors: {log1_stats.get('error_count', 0)}")
        report.append(f"  Log 2 Errors: {log2_stats.get('error_count', 0)}")
        
        report.append(f"  Log 1 Warnings: {log1_stats.get('warning_count', 0)}")
        report.append(f"  Log 2 Warnings: {log2_stats.get('warning_count', 0)}")
        
        report.append(f"  Log 1 Cache Hits: {log1_stats.get('cache_stats', {}).get('hits', 0)}")
        report.append(f"  Log 2 Cache Hits: {log2_stats.get('cache_stats', {}).get('hits', 0)}")
        
        report.append(f"  Log 1 Cache Misses: {log1_stats.get('cache_stats', {}).get('misses', 0)}")
        report.append(f"  Log 2 Cache Misses: {log2_stats.get('cache_stats', {}).get('misses', 0)}")
        
        report.append("")
        
        # Differences section
        differences = comparison_result.get('differences', {})
        if differences:
            report.append("KEY DIFFERENCES:")
            for key, diff in differences.items():
                if 'difference' in diff and diff['difference'] is not None:
                    report.append(f"  {key}: Log1={diff['log1']}, Log2={diff['log2']}, Difference={diff['difference']}")
                else:
                    report.append(f"  {key}: Log1={diff['log1']}, Log2={diff['log2']}")
            report.append("")
        
        # Convergence comparison
        conv_info = comparison_result.get('convergence_info', {})
        report.append("CONVERGENCE COMPARISON:")
        report.append(f"  Log 1 Converged: {conv_info.get('log1_converged', False)}")
        report.append(f"  Log 2 Converged: {conv_info.get('log2_converged', False)}")
        
        perf_comp = comparison_result.get('performance_comparison', {})
        report.append(f"  Log 1 Duration: {perf_comp.get('log1_duration', 'N/A')}")
        report.append(f"  Log 2 Duration: {perf_comp.get('log2_duration', 'N/A')}")
        report.append(f"  Faster Log: {perf_comp.get('faster_log', 'N/A')}")
        
        return "\n".join(report)
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str
    iteration: Optional[int] = None
    step: Optional[str] = None
    type_info: Optional[Dict[str, Any]] = None


@dataclass
class EMIteration:
    number: int
    start_time: datetime
    end_time: datetime
    log_entries: List[LogEntry]
    log_likelihood: Optional[float] = None
    change_in_likelihood: Optional[float] = None
    type_probabilities: Optional[List[float]] = None


class EstimationLogParser:
    """
    Parses estimation logs and provides structured access to the data.
    """
    
    def __init__(self):
        self.timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
        self.level_pattern = re.compile(r' - (INFO|DEBUG|WARNING|ERROR|CRITICAL) - ')
        self.em_iteration_pattern = re.compile(r'--- EM Iteration (\d+)/(\d+) ---')
        self.log_likelihood_pattern = re.compile(r'Log-Likelihood: ([\d.-]+)')
        self.change_likelihood_pattern = re.compile(r'Change in log-likelihood: ([\d.-]+)')
        self.type_probabilities_pattern = re.compile(r'Type probabilities: \[([^\]]+)\]')
        self.convergence_pattern = re.compile(r'EM algorithm converged after (\d+) iterations')
        self.bellman_converge_pattern = re.compile(r'Bellman equation converged for type (\d+) in (\d+) iterations')
        self.cache_pattern = re.compile(r'\[Likelihood\] (Cache HIT|Cache MISS) for agent_type=(\d+)')
        self.e_step_pattern = re.compile(r'Running E-step...')
        self.m_step_pattern = re.compile(r'Running M-step...')
        
    def parse_log_file(self, file_path: str) -> List[LogEntry]:
        """
        Parse a log file and return a list of LogEntry objects.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into lines
        lines = content.strip().split('\n')
        log_entries = []
        
        current_iteration = None
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract timestamp
            timestamp_match = self.timestamp_pattern.search(line)
            if not timestamp_match:
                continue  # Skip lines that don't have timestamps
            
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            except ValueError:
                continue  # Skip lines with invalid timestamps
            
            # Extract log level
            level_match = self.level_pattern.search(line)
            if level_match:
                level = level_match.group(1)
            else:
                level = 'INFO'  # Default level
            
            # Get the actual message part
            message_start = line.find(' - ', timestamp_match.end())
            if message_start != -1:
                message = line[message_start + 3:]  # Skip ' - ' part
            else:
                message = line[timestamp_match.end():].strip()
            
            # Check for EM iteration start - this resets the iteration counter
            iteration_match = self.em_iteration_pattern.search(message)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
            
            # Check for E-step start
            if self.e_step_pattern.search(message):
                current_step = 'E-step'
            
            # Check for M-step start
            if self.m_step_pattern.search(message):
                current_step = 'M-step'
            
            # Check for EM algorithm convergence to reset iteration tracking
            if 'EM algorithm converged' in message:
                current_iteration = None
                current_step = None
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=timestamp,
                level=level,
                message=message,
                iteration=current_iteration,
                step=current_step
            )
            
            # Extract additional information if available
            log_likelihood_match = self.log_likelihood_pattern.search(message)
            if log_likelihood_match:
                log_entry.type_info = log_entry.type_info or {}
                log_entry.type_info['log_likelihood'] = float(log_likelihood_match.group(1))
            
            change_likelihood_match = self.change_likelihood_pattern.search(message)
            if change_likelihood_match:
                log_entry.type_info = log_entry.type_info or {}
                log_entry.type_info['change_in_likelihood'] = float(change_likelihood_match.group(1))
            
            type_probabilities_match = self.type_probabilities_pattern.search(message)
            if type_probabilities_match:
                probabilities_str = type_probabilities_match.group(1)
                probabilities = [float(x.strip()) for x in probabilities_str.split()]
                log_entry.type_info = log_entry.type_info or {}
                log_entry.type_info['type_probabilities'] = probabilities
            
            log_entries.append(log_entry)
        
        return log_entries
    
    def group_by_iterations(self, log_entries: List[LogEntry]) -> List[EMIteration]:
        """
        Group log entries by EM iteration.
        """
        iterations = []
        current_iteration_data = None
        
        for entry in log_entries:
            # Check if this entry starts a new iteration
            if "--- EM Iteration" in entry.message and " ---" in entry.message:
                iteration_match = self.em_iteration_pattern.search(entry.message)
                if iteration_match:
                    iteration_num = int(iteration_match.group(1))
                    
                    # If we have a current iteration in progress, save it
                    if current_iteration_data is not None:
                        current_iteration_data.end_time = current_iteration_data.log_entries[-1].timestamp
                        iterations.append(current_iteration_data)
                    
                    # Start a new iteration
                    current_iteration_data = EMIteration(
                        number=iteration_num,
                        start_time=entry.timestamp,
                        end_time=entry.timestamp,
                        log_entries=[entry],  # Include the start entry
                        log_likelihood=None,
                        change_in_likelihood=None,
                        type_probabilities=None
                    )
                    continue  # Continue to next entry
            
            # If we're in an iteration, add the entry to it
            if current_iteration_data is not None:
                current_iteration_data.log_entries.append(entry)
                
                # Update iteration statistics
                if entry.type_info:
                    if 'log_likelihood' in entry.type_info:
                        current_iteration_data.log_likelihood = entry.type_info['log_likelihood']
                    if 'change_in_likelihood' in entry.type_info:
                        current_iteration_data.change_in_likelihood = entry.type_info['change_in_likelihood']
                    if 'type_probabilities' in entry.type_info:
                        current_iteration_data.type_probabilities = entry.type_info['type_probabilities']
                
                # Update end time
                if entry.timestamp > current_iteration_data.end_time:
                    current_iteration_data.end_time = entry.timestamp
            else:
                # If not in an iteration, this could be pre-iteration or post-iteration entries
                # Add to a temporary container that will be merged later if needed
                pass
        
        # Add the last iteration if it exists
        if current_iteration_data is not None:
            current_iteration_data.end_time = current_iteration_data.log_entries[-1].timestamp
            iterations.append(current_iteration_data)
        
        return iterations
    
    def extract_statistics(self, log_entries: List[LogEntry]) -> Dict[str, Any]:
        """
        Extract overall statistics from the log entries.
        """
        stats = {
            'total_entries': len(log_entries),
            'start_time': log_entries[0].timestamp if log_entries else None,
            'end_time': log_entries[-1].timestamp if log_entries else None,
            'total_duration': None,
            'iterations_completed': 0,
            'error_count': 0,
            'warning_count': 0,
            'log_likelihoods': [],
            'type_probabilities': [],
            'cache_stats': {'hits': 0, 'misses': 0}
        }
        
        if stats['start_time'] and stats['end_time']:
            stats['total_duration'] = stats['end_time'] - stats['start_time']
        
        for entry in log_entries:
            if entry.level == 'ERROR':
                stats['error_count'] += 1
            elif entry.level == 'WARNING':
                stats['warning_count'] += 1
                
            if entry.type_info:
                if 'log_likelihood' in entry.type_info:
                    stats['log_likelihoods'].append({
                        'iteration': entry.iteration,
                        'value': entry.type_info['log_likelihood'],
                        'timestamp': entry.timestamp
                    })
                
                if 'type_probabilities' in entry.type_info:
                    stats['type_probabilities'].append({
                        'iteration': entry.iteration,
                        'values': entry.type_info['type_probabilities'],
                        'timestamp': entry.timestamp
                    })
            
            # Check for cache hits/misses
            cache_match = self.cache_pattern.search(entry.message)
            if cache_match:
                cache_status = cache_match.group(1)
                if cache_status == 'Cache HIT':
                    stats['cache_stats']['hits'] += 1
                else:
                    stats['cache_stats']['misses'] += 1
        
        # Count completed iterations
        if log_entries:
            iterations = [entry.iteration for entry in log_entries if entry.iteration is not None]
            if iterations:
                stats['iterations_completed'] = max(iterations) if iterations else 0
        
        return stats

    def compare_logs(self, log_entries1: List[LogEntry], log_entries2: List[LogEntry]) -> Dict[str, Any]:
        """
        Compare two log files and extract key differences.
        """
        stats1 = self.extract_statistics(log_entries1)
        stats2 = self.extract_statistics(log_entries2)
        
        comparison = {
            'log1_stats': stats1,
            'log2_stats': stats2,
            'differences': {}
        }
        
        # Compare key metrics
        diff_fields = [
            'total_entries', 'total_duration', 'iterations_completed', 
            'error_count', 'warning_count', 'cache_stats'
        ]
        
        for field in diff_fields:
            val1 = stats1.get(field)
            val2 = stats2.get(field)
            
            if val1 != val2:
                comparison['differences'][field] = {
                    'log1': val1,
                    'log2': val2,
                    'difference': None
                }
                
                # Calculate difference for numeric values
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    comparison['differences'][field]['difference'] = val2 - val1
        
        # Compare log likelihood progression
        ll1 = stats1.get('log_likelihoods', [])
        ll2 = stats2.get('log_likelihoods', [])
        
        if ll1 and ll2:
            # Get the final log likelihoods
            final_ll1 = ll1[-1]['value'] if ll1 else None
            final_ll2 = ll2[-1]['value'] if ll2 else None
            
            if final_ll1 and final_ll2:
                comparison['differences']['final_log_likelihood'] = {
                    'log1': final_ll1,
                    'log2': final_ll2,
                    'difference': final_ll2 - final_ll1
                }
        
        # Compare convergence
        comparison['convergence_info'] = {
            'log1_converged': stats1.get('iterations_completed', 0) > 0,
            'log2_converged': stats2.get('iterations_completed', 0) > 0,
        }
        
        # Compare performance metrics
        comparison['performance_comparison'] = {
            'log1_duration': str(stats1.get('total_duration', 'N/A')),
            'log2_duration': str(stats2.get('total_duration', 'N/A')),
            'faster_log': 'log1' if stats1.get('total_duration') and stats2.get('total_duration') and stats1['total_duration'] < stats2['total_duration'] else 'log2' if stats1.get('total_duration') and stats2.get('total_duration') else 'unknown'
        }
        
        return comparison
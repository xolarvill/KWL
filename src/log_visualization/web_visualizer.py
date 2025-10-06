import json
from datetime import datetime
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory
from .log_parser import EstimationLogParser, LogEntry, EMIteration
from .log_visualizer import LogVisualizer, FilterOptions
import os


class WebLogVisualizer:
    """
    Web-based log visualization interface using Flask.
    """
    
    def __init__(self, log_dir: str = "progress/log"):
        self.app = Flask(__name__)
        self.log_dir = log_dir
        self.parser = EstimationLogParser()
        self.visualizer = LogVisualizer(self.parser)
        self.current_entries = []
        self.current_iterations = []
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            return render_template('log_visualizer.html')
        
        @self.app.route('/logs')
        def get_log_files():
            """Get list of available log files."""
            log_files = []
            if os.path.exists(self.log_dir):
                for file in os.listdir(self.log_dir):
                    if file.endswith('.log'):
                        log_files.append(file)
            
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)), reverse=True)
            return jsonify(log_files)
        
        @self.app.route('/load_log', methods=['POST'])
        def load_log():
            """Load and parse a specific log file."""
            try:
                filename = request.json.get('filename')
                filepath = os.path.join(self.log_dir, filename)
                
                if not os.path.exists(filepath):
                    return jsonify({'error': 'File not found'}), 404
                
                self.current_entries = self.parser.parse_log_file(filepath)
                self.current_iterations = self.parser.group_by_iterations(self.current_entries)
                
                # Return basic statistics
                stats = self.parser.extract_statistics(self.current_entries)
                
                # Convert non-serializable objects to strings recursively
                serializable_stats = self._make_serializable(stats)
                
                return jsonify({
                    'success': True,
                    'stats': serializable_stats,
                    'iterations_count': len(self.current_iterations)
                })
            except Exception as e:
                import traceback
                return jsonify({'error': str(e) + '\\n' + traceback.format_exc()}), 500
        
        @self.app.route('/filter_logs', methods=['POST'])
        def filter_logs():
            """Filter loaded log entries based on user options."""
            try:
                filters_data = request.json.get('filters', {})
                
                # Create filter options object
                filters = FilterOptions(
                    min_level=filters_data.get('min_level', 'INFO'),
                    iteration_filter=filters_data.get('iteration_filter'),
                    step_filter=filters_data.get('step_filter'),
                    search_text=filters_data.get('search_text', ''),
                    show_cache_operations=filters_data.get('show_cache_operations', True),
                    show_bellman_operations=filters_data.get('show_bellman_operations', True),
                )
                
                filtered_entries = self.visualizer.filter_entries(self.current_entries, filters)
                
                # Format entries for JSON response
                formatted_entries = []
                for entry in filtered_entries:
                    formatted_entries.append({
                        'timestamp': entry.timestamp.isoformat(),
                        'level': entry.level,
                        'message': entry.message,
                        'iteration': entry.iteration,
                        'step': entry.step
                    })
                
                return jsonify({
                    'entries': formatted_entries,
                    'count': len(formatted_entries)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/summary')
        def get_summary():
            """Get summary report of loaded logs."""
            try:
                summary = self.visualizer.generate_summary_report(self.current_entries, self.current_iterations)
                return jsonify({'summary': summary})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/performance')
        def get_performance():
            """Get performance analysis of loaded logs."""
            try:
                performance = self.visualizer.print_performance_analysis(self.current_iterations)
                return jsonify({'performance': performance})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/timeline')
        def get_timeline():
            """Get timeline view of loaded logs."""
            try:
                timeline = self.visualizer.print_timeline_view(self.current_entries)
                return jsonify({'timeline': timeline})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/assets/<path:filename>')
        def assets(filename):
            """Serve static assets."""
            return send_from_directory(os.path.join(os.path.dirname(__file__), 'assets'), filename)

    def _make_serializable(self, obj):
        """Convert non-serializable objects to strings recursively."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'total_seconds'):  # timedelta and similar
            return str(obj)
        else:
            # For any other object that might not be serializable
            return str(obj)
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the web application."""
        self.app.run(host=host, port=port, debug=debug)


# HTML template for the log visualizer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Estimation Log Visualizer</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            background-color: #1e1e1e;
            color: #dcdcdc;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: #4ec9b0;
            border-bottom: 2px solid #4ec9b0;
            padding-bottom: 10px;
        }
        
        .controls {
            background: #2d2d30;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .control-group {
            margin: 10px 0;
        }
        
        label {
            display: inline-block;
            width: 150px;
            color: #9cdcfe;
        }
        
        input, select, button {
            background: #3c3c40;
            color: #dcdcdc;
            border: 1px solid #454545;
            padding: 5px 10px;
            border-radius: 3px;
        }
        
        button {
            background: #007acc;
            color: white;
            border: none;
            padding: 8px 15px;
            cursor: pointer;
        }
        
        button:hover {
            background: #005a9e;
        }
        
        .log-options {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .log-options select {
            flex: 1;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 10px;
            border-bottom: 1px solid #454545;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background: #2d2d30;
            border: 1px solid #454545;
            border-bottom: none;
            margin-right: 5px;
        }
        
        .tab.active {
            background: #007acc;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .log-container {
            background: #1e1e1e;
            border: 1px solid #454545;
            border-radius: 5px;
            padding: 15px;
            height: 500px;
            overflow-y: auto;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 14px;
            white-space: pre-wrap;
            line-height: 1.4;
        }
        
        .log-entry {
            margin-bottom: 5px;
        }
        
        .timestamp {
            color: #d19a66;
        }
        
        .level-INFO {
            color: #dcdcaa;
        }
        
        .level-ERROR {
            color: #f44747;
        }
        
        .level-WARNING {
            color: #d19a66;
        }
        
        .level-DEBUG {
            color: #b5cea8;
        }
        
        .level-CRITICAL {
            color: #ff0000;
            font-weight: bold;
        }
        
        .iteration {
            color: #c586c0;
        }
        
        .step {
            color: #9cdcfe;
        }
        
        .cache-hit {
            color: #89d185;
        }
        
        .cache-miss {
            color: #75beff;
        }
        
        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .stat-box {
            background: #2d2d30;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4ec9b0;
        }
        
        .stat-label {
            color: #9cdcfe;
            font-size: 14px;
        }
        
        .loading {
            text-align: center;
            color: #9cdcfe;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Estimation Log Visualizer</h1>
        
        <div class="controls">
            <div class="log-options">
                <select id="logFileSelect">
                    <option value="">Select a log file...</option>
                </select>
                <button onclick="loadLogFile()">Load Log</button>
                <button onclick="refreshLogList()">Refresh List</button>
            </div>
            
            <div class="control-group">
                <label for="minLevel">Min Level:</label>
                <select id="minLevel">
                    <option value="DEBUG">DEBUG</option>
                    <option value="INFO" selected>INFO</option>
                    <option value="WARNING">WARNING</option>
                    <option value="ERROR">ERROR</option>
                    <option value="CRITICAL">CRITICAL</option>
                </select>
                
                <label for="searchText">Search:</label>
                <input type="text" id="searchText" placeholder="Enter search text...">
                
                <button onclick="applyFilters()">Apply Filters</button>
            </div>
            
            <div class="control-group">
                <label>
                    <input type="checkbox" id="showCacheOps" checked> Show Cache Operations
                </label>
                <label>
                    <input type="checkbox" id="showBellmanOps" checked> Show Bellman Operations
                </label>
            </div>
        </div>
        
        <div class="stats-panel" id="statsPanel" style="display: none;">
            <div class="stat-box">
                <div class="stat-value" id="totalEntries">0</div>
                <div class="stat-label">Total Entries</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="iterationsCount">0</div>
                <div class="stat-label">Iterations</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="errorCount">0</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="warningCount">0</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="cacheHits">0</div>
                <div class="stat-label">Cache Hits</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="cacheMisses">0</div>
                <div class="stat-label">Cache Misses</div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('summary')">Summary</div>
            <div class="tab" onclick="switchTab('timeline')">Timeline</div>
            <div class="tab" onclick="switchTab('performance')">Performance</div>
            <div class="tab" onclick="switchTab('filtered')">Filtered Logs</div>
        </div>
        
        <div class="tab-content active" id="summary">
            <div class="log-container" id="summaryContent">Select a log file to view summary...</div>
        </div>
        
        <div class="tab-content" id="timeline">
            <div class="log-container" id="timelineContent">Select a log file to view timeline...</div>
        </div>
        
        <div class="tab-content" id="performance">
            <div class="log-container" id="performanceContent">Select a log file to view performance...</div>
        </div>
        
        <div class="tab-content" id="filtered">
            <div class="log-container" id="filteredContent">Filtered logs will appear here...</div>
        </div>
    </div>
    
    <script>
        let currentLogEntries = [];
        let currentIterations = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            refreshLogList();
        });
        
        function refreshLogList() {
            fetch('/logs')
                .then(response => response.json())
                .then(data => {
                    const select = document.getElementById('logFileSelect');
                    select.innerHTML = '<option value="">Select a log file...</option>';
                    
                    data.forEach(file => {
                        const option = document.createElement('option');
                        option.value = file;
                        option.textContent = file;
                        select.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('Error loading log files:', error);
                    alert('Error loading log files: ' + error.message);
                });
        }
        
        function loadLogFile() {
            const filename = document.getElementById('logFileSelect').value;
            if (!filename) {
                alert('Please select a log file');
                return;
            }
            
            // Show loading
            document.getElementById('summaryContent').textContent = 'Loading...';
            document.getElementById('timelineContent').textContent = 'Loading...';
            document.getElementById('performanceContent').textContent = 'Loading...';
            document.getElementById('filteredContent').textContent = 'Loading...';
            
            fetch('/load_log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({filename: filename})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('statsPanel').style.display = 'grid';
                updateStats(data.stats);
                
                // Load summary, timeline, and performance
                refreshSummary();
                refreshTimeline();
                refreshPerformance();
                
                // Initially show all entries in filtered view
                refreshFilteredView();
            })
            .catch(error => {
                console.error('Error loading log:', error);
                alert('Error loading log: ' + error.message);
            });
        }
        
        function updateStats(stats) {
            document.getElementById('totalEntries').textContent = stats.total_entries || 0;
            document.getElementById('iterationsCount').textContent = stats.iterations_completed || 0;
            document.getElementById('errorCount').textContent = stats.error_count || 0;
            document.getElementById('warningCount').textContent = stats.warning_count || 0;
            document.getElementById('cacheHits').textContent = stats.cache_stats?.hits || 0;
            document.getElementById('cacheMisses').textContent = stats.cache_stats?.misses || 0;
        }
        
        function applyFilters() {
            const filters = {
                min_level: document.getElementById('minLevel').value,
                search_text: document.getElementById('searchText').value,
                show_cache_operations: document.getElementById('showCacheOps').checked,
                show_bellman_operations: document.getElementById('showBellmanOps').checked
            };
            
            fetch('/filter_logs', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({filters: filters})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                displayFilteredLogs(data.entries);
            })
            .catch(error => {
                console.error('Error filtering logs:', error);
                alert('Error filtering logs: ' + error.message);
            });
        }
        
        function displayFilteredLogs(entries) {
            const container = document.getElementById('filteredContent');
            container.innerHTML = '';
            
            if (entries.length === 0) {
                container.textContent = 'No matching log entries found.';
                return;
            }
            
            entries.forEach(entry => {
                const entryDiv = document.createElement('div');
                entryDiv.className = 'log-entry';
                
                const timestamp = document.createElement('span');
                timestamp.className = 'timestamp';
                timestamp.textContent = `[${new Date(entry.timestamp).toLocaleTimeString()}] `;
                
                const level = document.createElement('span');
                level.className = `level-${entry.level}`;
                level.textContent = `${entry.level} `;
                
                const iteration = document.createElement('span');
                if (entry.iteration) {
                    iteration.className = 'iteration';
                    iteration.textContent = `(Iter ${entry.iteration}) `;
                }
                
                const step = document.createElement('span');
                if (entry.step) {
                    step.className = 'step';
                    step.textContent = `[${entry.step}] `;
                }
                
                const message = document.createElement('span');
                message.textContent = entry.message;
                
                // Add special styling for cache operations
                if (entry.message.includes('Cache HIT')) {
                    message.className = 'cache-hit';
                } else if (entry.message.includes('Cache MISS')) {
                    message.className = 'cache-miss';
                }
                
                entryDiv.appendChild(timestamp);
                entryDiv.appendChild(level);
                if (entry.iteration) entryDiv.appendChild(iteration);
                if (entry.step) entryDiv.appendChild(step);
                entryDiv.appendChild(message);
                
                container.appendChild(entryDiv);
            });
        }
        
        function refreshSummary() {
            fetch('/summary')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('summaryContent').textContent = 'Error: ' + data.error;
                    } else {
                        document.getElementById('summaryContent').textContent = data.summary;
                    }
                })
                .catch(error => {
                    document.getElementById('summaryContent').textContent = 'Error loading summary: ' + error.message;
                });
        }
        
        function refreshTimeline() {
            fetch('/timeline')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('timelineContent').textContent = 'Error: ' + data.error;
                    } else {
                        document.getElementById('timelineContent').textContent = data.timeline;
                    }
                })
                .catch(error => {
                    document.getElementById('timelineContent').textContent = 'Error loading timeline: ' + error.message;
                });
        }
        
        function refreshPerformance() {
            fetch('/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('performanceContent').textContent = 'Error: ' + data.error;
                    } else {
                        document.getElementById('performanceContent').textContent = data.performance;
                    }
                })
                .catch(error => {
                    document.getElementById('performanceContent').textContent = 'Error loading performance: ' + error.message;
                });
        }
        
        function refreshFilteredView() {
            // Apply current filters
            applyFilters();
        }
        
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

def create_templates():
    """Create template files if they don't exist."""
    import os
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    with open(os.path.join(template_dir, 'log_visualizer.html'), 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)

# Create templates directory and save the template when module is imported
create_templates()
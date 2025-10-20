import os
import numpy as np
import logging

# ------ Logger setup ------
logger = logging.getLogger("TraceLoader")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
# --------------------------

class NetworkTrace:
    """
    ذخیره و مدیریت یک فایل trace شبکه
    """
    def __init__(self, trace_id, timestamps, throughputs, metadata=None):
        self.trace_id = trace_id
        self.timestamps = np.array(timestamps, dtype=np.float64)
        self.throughputs = np.array(throughputs, dtype=np.float64)
        self.metadata = metadata if metadata is not None else {}
        
        if not len(self.timestamps) == len(self.throughputs):
            raise ValueError("Timestamps and throughputs must have the same length")
        
        self.duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0.0

    def get_download_time(self, chunk_size_bytes, start_time_s):
        """
        شبیه‌سازی زمان دانلود بر اساس بایت
        """
        if not len(self.timestamps) > 0:
            return 1.0
        
        chunk_size_kbit = (chunk_size_bytes * 8.0) / 1000.0
        downloaded_kbit = 0.0
        download_time_s = 0.0
        
        try:
            start_idx = np.searchsorted(self.timestamps, start_time_s)
            if start_idx == len(self.timestamps):
                start_idx -= 1
        except IndexError:
            start_idx = 0
            
        current_time = start_time_s
        
        for i in range(start_idx, len(self.timestamps)):
            if downloaded_kbit >= chunk_size_kbit:
                break
                
            time_diff_s = (self.timestamps[i+1] - max(current_time, self.timestamps[i])) if i + 1 < len(self.timestamps) else 0.1
            throughput_kbps = self.throughputs[i]
            
            downloaded_kbit += throughput_kbps * time_diff_s
            download_time_s += time_diff_s
            current_time += time_diff_s
            
            if download_time_s > 30.0:
                break
        
        if downloaded_kbit < chunk_size_kbit:
            remaining_kbit = chunk_size_kbit - downloaded_kbit
            last_throughput = self.throughputs[-1] if self.throughputs[-1] > 0 else 1.0
            download_time_s += remaining_kbit / last_throughput
            
        return download_time_s

    def get_throughput(self, time_s):
        """
        دریافت throughput در یک زمان مشخص
        """
        if not len(self.timestamps) > 0: return 0
        idx = np.searchsorted(self.timestamps, time_s)
        if idx == 0: return self.throughputs[0]
        if idx >= len(self.timestamps): return self.throughputs[-1]
        return self.throughputs[idx - 1]


class TraceLoader:
    """
    لود کردن فایل‌های trace از پوشه‌های cooked_traces
    """
    
    def __init__(self, trace_dir='data/network_traces/cooked_traces'):
        self.trace_dir = trace_dir
        self.trace_files = {'train': [], 'val': [], 'test': []}
        
        # for backward compatibility with scripts using .test_traces
        self.train_traces = self.trace_files['train']
        self.val_traces = self.trace_files['val']
        self.test_traces = self.trace_files['test']

        self.traces = {'train': {}, 'val': {}, 'test': {}}
        self._load_traces()

    def _load_traces(self):
        """
        لود کردن فایل‌های trace و تقسیم‌بندی آن‌ها
        """
        if not os.path.exists(self.trace_dir):
            logger.error(f"Trace directory not found: {self.trace_dir}")
            return
            
        logger.info(f"Loading traces from: {self.trace_dir}")
        
        all_files = []
        # First, check for split-specific subdirectories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.trace_dir, f"cooked_{split}_traces")
            if os.path.exists(split_dir):
                 for file_name in os.listdir(split_dir):
                    if not file_name.endswith('.log'):
                        self.trace_files[split].append(os.path.join(split_dir, file_name))
        
        # If any split is empty, load from the main directory and assign
        if not any(self.trace_files.values()):
            logger.warning("No split-specific folders found. Loading all from base directory.")
            all_files = [os.path.join(self.trace_dir, f) for f in os.listdir(self.trace_dir) if not f.endswith('.log')]
            np.random.shuffle(all_files)
            train_end = int(0.7 * len(all_files))
            val_end = int(0.85 * len(all_files))
            self.trace_files['train'] = all_files[:train_end]
            self.trace_files['val'] = all_files[train_end:val_end]
            self.trace_files['test'] = all_files[val_end:]

        logger.info(f"✓ Loaded {len(self.trace_files['train'])} train, "
                    f"{len(self.trace_files['val'])} val, "
                    f"{len(self.trace_files['test'])} test traces")

    def load_trace_file(self, file_path):
        """
        خواندن فایل trace به صورت آرایه‌های numpy
        """
        timestamps, throughputs = [], []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.split()
                        timestamps.append(float(parts[0]))
                        throughputs.append(float(parts[1]))
                    except (IndexError, ValueError):
                        continue
            
            if not timestamps:
                logger.warning(f"No valid data in trace: {file_path}")
                return None
                
            return NetworkTrace(os.path.basename(file_path), timestamps, throughputs)
        except Exception as e:
            logger.error(f"Error loading trace {file_path}: {e}")
            return None

    def get_trace_files(self, split='train'):
        """
        ✅ این متد لیست فایل‌های trace برای یک مجموعه (split) را برمی‌گرداند.
        """
        return self.trace_files.get(split, [])

    def sample_trace(self, split='train'):
        """
        نمونه‌گیری تصادفی یک trace از مجموعه مشخص شده
        """
        if not self.trace_files[split]:
            logger.error(f"No traces found for split: {split}")
            return None
            
        trace_path = np.random.choice(self.trace_files[split])
        
        if trace_path not in self.traces[split]:
            self.traces[split][trace_path] = self.load_trace_file(trace_path)
            
        return self.traces[split][trace_path]
    
    def get_trace_by_id(self, trace_id, split='train'):
        """
        گرفتن یک trace بر اساس نام فایل (ID)
        """
        for trace_path in self.trace_files[split]:
            if os.path.basename(trace_path) == trace_id:
                if trace_path not in self.traces[split]:
                    self.traces[split][trace_path] = self.load_trace_file(trace_path)
                return self.traces[split][trace_path]
        
        logger.error(f"Trace ID not found in split '{split}': {trace_id}")
        return None

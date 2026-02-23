import sys
import datetime
from collections import deque
import os, torch, json


class ConsoleLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_stream = open(log_file, 'a', encoding='utf-8')
        
        class CustomStdout:
            def __init__(self, original, log_stream):
                self.original = original
                self.log_stream = log_stream
            
            def write(self, message):
                self.original.write(message)
                self.log_stream.write(message)
                self.log_stream.flush()
            
            def flush(self):
                self.original.flush()
                self.log_stream.flush()
            
            def isatty(self):
                return hasattr(self.original, 'isatty') and self.original.isatty()
        
        self.custom_stdout = CustomStdout(self.original_stdout, self.log_stream)
        self.custom_stderr = CustomStdout(self.original_stderr, self.log_stream)
        
    def start_logging(self):
        sys.stdout = self.custom_stdout
        sys.stderr = self.custom_stderr
        print(f"Logging started, log file: {self.log_file}")
        
    def stop_logging(self):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if not self.log_stream.closed:
            self.log_stream.close()
        print(f"Logging ended, log file saved to: {self.log_file}")

console_logger = None


def setup_console_logger(log_dir, model_name):
    global console_logger
    
    console_log_dir = os.path.join(log_dir, 'console_logs')
    
    log_filename = os.path.join(console_log_dir, 
                               f'{model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    console_logger = ConsoleLogger(log_filename)

    console_logger.start_logging()
    
    return console_logger


def close_console_logger():
    global console_logger
    if console_logger is not None:
        console_logger.stop_logging()
        console_logger = None
        
# Global frontier data structures
FRONTIER = []
FRONTIER_ARCHIVE = []
FRONTIER_CAP = 20

def dominates(a, b):
    """
    Determine if a dominates b according to the new rules:
    - First judge by metrics: a.psnr >= b.psnr, a.ssim >= b.ssim, a.lpips <= b.lpips
    - As long as a is non-inferior in all metrics and at least one is strictly better, it is considered dominant
    - If the three metrics are basically equal within tolerance, use score as tie-breaker (higher score is better)
    - lpips of None is treated as inf; if both ambos have lpips of None, judge by score
    - If val_only_psnr mode is enabled, only compare PSNR
    """
    a_psnr, a_ssim, a_lpips = a['psnr'], a['ssim'], a['lpips']
    b_psnr, b_ssim, b_lpips = b['psnr'], b['ssim'], b['lpips']

    # Check if val_only_psnr mode is enabled
    a_val_only_psnr = a.get('val_only_psnr', False)
    b_val_only_psnr = b.get('val_only_psnr', False)
    val_only_psnr = a_val_only_psnr or b_val_only_psnr  # If either side enables, only compare PSNR

    # None -> inf handling (only when not in val_only_psnr mode)
    if not val_only_psnr:
        if a_lpips is None: a_lpips = float('inf')
        if b_lpips is None: b_lpips = float('inf')

    a_score = a.get('score')
    b_score = b.get('score')

    # Tolerance parameters
    tol1= 0.5
    tol2 = 0.005
    tol3 = 0.015

    # Determine non-inferior and at least one strictly better
    if val_only_psnr:
        # In val_only_psnr mode, only compare PSNR
        non_worse = (a_psnr >= b_psnr)
        strictly_better = (a_psnr > b_psnr)
    else:
        # In normal mode, compare PSNR, SSIM and LPIPS
        non_worse = (a_psnr >= b_psnr) and (a_ssim >= b_ssim) and (a_lpips <= b_lpips)
        strictly_better = (a_psnr > b_psnr) or (a_ssim > b_ssim) or (a_lpips < b_lpips)

    if non_worse and strictly_better:
        return True

    # If metrics are basically equal, use score as tie-breaker
    if val_only_psnr:
        # In val_only_psnr mode, only compare PSNR
        eq_all = (abs(a_psnr - b_psnr) <= tol1)
    else:
        # In normal mode, compare PSNR, SSIM and LPIPS
        eq_all = (abs(a_psnr - b_psnr) <= tol1) and (abs(a_ssim - b_ssim) <= tol2) and (abs(a_lpips - b_lpips) <= tol3)

    if eq_all:
        # Only compare by score when both sides have score
        if (a_score is not None) and (b_score is not None):
            return a_score > b_score
        else:
            return False

    # Other cases (including trade-off relationships) do not dominate
    return False

class FixedRangePSNRNormalizer:
    def __init__(self, min_psnr=0.0, max_psnr=40.0):
        """
        Fixed range PSNR normalizer

        Args:
            min_psnr: Expected minimum PSNR value
            max_psnr: Expected maximum PSNR value
        """
        self.min_val = min_psnr
        self.max_val = max_psnr

    def update(self, val):
        """Fixed range normalization does not need to update state, but maintains interface consistency"""
        pass

    def norm(self, val):
        """
        Normalize PSNR value to [0,1] range

        Args:
            val: PSNR value

        Returns:
            Normalized value, range [0,1]
        """
        current_val = float(val)

        # Handle out of range cases
        if current_val <= self.min_val:
            return 0.0
        elif current_val >= self.max_val:
            return 1.0

        # Linear normalization
        normalized = (current_val - self.min_val) / (self.max_val - self.min_val)
        return max(0.0, min(1.0, normalized))

# Global instance: use the actual range you observed
psnr_normalizer = FixedRangePSNRNormalizer(min_psnr=5.0, max_psnr=25.0)

def normalize_metrics(psnr, ssim, lpips, use_lpips=True, psnr_normer=None):
    """
    Normalize three metrics to the same scale [0,1], return (psnr_norm, ssim_norm, lpips_norm)
    - PSNR: Use dynamic range normalization
    - SSIM: Direct value [0,1]
    - LPIPS: Smaller is better, take 1 - clamp(lpips, 0, 1)
    """
    if psnr_normer is None:
        psnr_normer = psnr_normalizer
    psnr_norm = psnr_normer.norm(float(psnr))

    ssim_norm = max(0.0, min(1.0, float(ssim)))

    if use_lpips:
        lpips_clamped = max(0.0, min(1.0, float(lpips)))
        lpips_norm = 1.0 - lpips_clamped
        return psnr_norm, ssim_norm, lpips_norm
    else:
        return psnr_norm, ssim_norm, None

def compute_score_normalized(psnr, ssim, lpips, use_lpips=True,
                             w1=0.4, w2=0.4, w3=0.2, psnr_normer=None):
    psnr_norm, ssim_norm, lpips_norm = normalize_metrics(psnr, ssim, lpips, use_lpips, psnr_normer)
    if use_lpips:
        return w1 * psnr_norm + w2 * ssim_norm + w3 * lpips_norm
    else:
        return w1 * psnr_norm + w2 * ssim_norm

def _get_dirs(frontier_root, model_name):
    # Ensure root directory exists; if frontier_root is not given, use default './frontier_saves'
    if frontier_root is None:
        frontier_root = './frontier_saves'
    base = os.path.join(frontier_root, model_name)
    weights = os.path.join(base, 'weights')
    archives = os.path.join(base, 'archives')
    os.makedirs(weights, exist_ok=True)
    os.makedirs(archives, exist_ok=True)
    return base, weights, archives

def save_frontier_model(candidate, frontier_root, model_name):
    _, weights_dir, _ = _get_dirs(frontier_root, model_name)
    step = candidate['step']
    path = os.path.join(weights_dir, f'frontier_{step}.pth')
    # Save complete candidate information, including all required fields
    save_data = {
        'step': step,
        'max_psnr': candidate.get('max_psnr'),
        'max_ssim': candidate.get('max_ssim'),
        'min_lpips': candidate.get('min_lpips'),
        'ssims': candidate.get('ssims'),
        'psnrs': candidate.get('psnrs'),
        'lpips_list': candidate.get('lpips_list'),
        'losses': candidate.get('losses'),
        'model': candidate.get('model')
    }
    torch.save(save_data, path)
    return path

def save_frontier_archive(candidate, frontier_root, model_name):
    _, _, archives_dir = _get_dirs(frontier_root, model_name)
    step = candidate['step']
    data = {
        'step': step,
        'psnr': candidate.get('psnr'),
        'ssim': candidate.get('ssim'),
        'lpips': candidate.get('lpips'),
        'score': candidate.get('score')
    }
    path = os.path.join(archives_dir, f'frontier_{step}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path

def update_frontier(candidate, use_lpips, frontier_root=None, model_name=None):
    """
    candidate: dict with keys ['step','psnr','ssim','lpips','state_dict','score'(optional)]
    frontier_root/model_name used to locate save directory
    Returns True if entered frontier, False if not entered
    """
    global FRONTIER, FRONTIER_ARCHIVE
    if model_name is None:
        # No model name, cannot categorize by directory, return False directly
        return False

    required = ['step','psnr','ssim','model']
    if any(k not in candidate for k in required):
        return False

    # 1) Calculate indices of items dominated by candidate
    dominated_indices = [idx for idx, f in enumerate(FRONTIER) if dominates(candidate, f)]

    # frontier not full: add directly
    if len(FRONTIER) < FRONTIER_CAP:
        FRONTIER.append(candidate)
        FRONTIER_ARCHIVE.append(candidate.copy())

        # Save model that entered frontier (if save directory is provided)
        if candidate.get('model') is not None:
            save_frontier_model(candidate, frontier_root, model_name)
            save_frontier_archive(candidate, frontier_root, model_name)

        return True

    # frontier is full
    if len(dominated_indices) == 0:
        # Frontier is full and candidate cannot dominate any existing item, do not enter according to your rules
        return False

    # 4) Among items dominated by candidate, select the one with lowest score to replace
    def item_score(e):
        s = e.get('score')
        if s is None:
            s = compute_score_normalized(e['psnr'], e['ssim'], e['lpips'],
                                         use_lpips, psnr_normer=psnr_normalizer)
            e['score'] = s
        return s

    worst_idx = min(dominated_indices, key=lambda i: item_score(FRONTIER[i]))

    # Perform replacement: replace FRONTIER[worst_idx] with candidate
    FRONTIER[worst_idx] = candidate
    FRONTIER_ARCHIVE.append(candidate.copy())

    # 4) Save model that entered frontier (if save directory is provided)
    if candidate.get('model') is not None:
        save_frontier_model(candidate, frontier_root, model_name)
        save_frontier_archive(candidate, frontier_root, model_name)

    return True


def finalize_frontier_selection(use_lpips, top_k=5, frontier_root=None, model_name=None):
    scored = []
    for e in FRONTIER_ARCHIVE:
        s = e.get('score')
        if s is None:
            s = compute_score_normalized(e['psnr'], e['ssim'], e['lpips'], use_lpips, psnr_normer=psnr_normalizer)
            e['score'] = s
        scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item[1] for item in scored[:top_k]]
    return top

def final_frontier_deploy(use_lpips, top_k=5, frontier_root=None, model_name=None):
    top = finalize_frontier_selection(use_lpips, top_k, frontier_root, model_name)
    for i, c in enumerate(top, 1):
        step = c['step']
        path = None
        if c.get('model') is not None:
            # Save final Top-K models to weights directory
            _, weights_dir, _ = _get_dirs(frontier_root, model_name)
            path = os.path.join(weights_dir, f'final_frontier_{i}_step{step}.pth')
            # Save complete candidate information, including all required fields
            save_data = {
                'step': step,
                'max_psnr': c.get('max_psnr'),
                'max_ssim': c.get('max_ssim'),
                'min_lpips': c.get('min_lpips'),
                'ssims': c.get('ssims'),
                'psnrs': c.get('psnrs'),
                'lpips_list': c.get('lpips_list'),
                'losses': c.get('losses'),
                'model': c.get('model')
            }
            torch.save(save_data, path)
        print(f"{i}. step={step}, PSNR={c['psnr']:.4f}, SSIM={c['ssim']:.4f}, LPIPS={c['lpips'] if use_lpips else 'NA'} -> final path: {path if path else 'N/A'}")
    return top

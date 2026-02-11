#!/usr/bin/env python
"""
Calculates median and IQR for experimental results for each recorded step
from gradient_flow_digits.py.

This script reads the *_distances.txt files from the specified log directory.
For each unique (loss_type, lr) pair, and for each step recorded in the
distance files, it computes aggregate statistics (median, Q1, Q3, IQR)
across all seeds.
"""
import argparse
from pathlib import Path
import re
import numpy as np
from collections import defaultdict

def parse_filename(filename_str: str):
    """
    Parses a filename to extract loss_type, lr, and seed.
    Expected format: images_{loss_type}_lr{lr_value}_seed{seed_value}_distances.txt
    """
    match = re.match(r"images_(.+)_lr([0-9.eE+-]+)_seed(\d+)_distances\.txt", Path(filename_str).name)
    if match:
        loss_type = match.group(1)
        lr_str = match.group(2)
        seed_str = match.group(3)
        try:
            lr_val = float(lr_str)
            # seed_val is parsed but not directly used in aggregation keys for this script's main purpose
            seed_val = int(seed_str)
            return loss_type, lr_val, seed_val
        except ValueError:
            print(f"Warning: Could not parse numeric lr/seed from filename parts: '{lr_str}', '{seed_str}' in '{filename_str}'")
            return None
    else:
        # print(f"Debug: Filename '{filename_str}' did not match expected pattern.") # Optional
        return None

def calculate_statistics(data_list: list[float]):
    """
    Calculates median, Q1, Q3, IQR, and count for a list of numbers.
    Returns (median, q1, q3, iqr, count).
    Returns NaNs for stats if data_list is empty.
    """
    count = len(data_list)
    if count == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    median_val = np.median(data_list)
    q1_val = np.percentile(data_list, 25)
    q3_val = np.percentile(data_list, 75)
    iqr_val = q3_val - q1_val
    return median_val, q1_val, q3_val, iqr_val, count

import argparse
from pathlib import Path
import re
import numpy as np
from collections import defaultdict

# Assume parse_filename is defined as previously:
def parse_filename(filename_str: str):
    """
    Parses a filename to extract loss_type, lr, and seed.
    Expected format: images_{loss_type}_lr{lr_value}_seed{seed_value}_distances.txt
    """
    match = re.match(r"images_(.+)_lr([0-9.eE+-]+)_seed(\d+)_distances\.txt", Path(filename_str).name)
    if match:
        loss_type = match.group(1)
        lr_str = match.group(2)
        seed_str = match.group(3)
        try:
            lr_val = float(lr_str)
            # seed_val is parsed but not directly used in aggregation keys for this script's main purpose
            _ = int(seed_str) # validate seed is int
            return loss_type, lr_val, _ # Return original lr_val as float
        except ValueError:
            print(f"Warning: Could not parse numeric lr/seed from filename parts: '{lr_str}', '{seed_str}' in '{filename_str}'")
            return None
    else:
        # print(f"Debug: Filename '{filename_str}' did not match expected pattern.")
        return None

def calculate_statistics(data_list: list[float]):
    """
    Calculates count, specified percentiles (P5, P10, Q1, Median, Q3, P90, P95), and IQR.
    Returns (p5, p10, q1, median, q3, p90, p95, iqr, count).
    Returns NaNs for stats if data_list is empty.
    """
    count = len(data_list)
    num_stats_plus_iqr = 8 # P5, P10, Q1, Median, Q3, P90, P95, IQR

    if count == 0:
        return tuple([np.nan] * num_stats_plus_iqr) + (0,)

    # Percentiles to calculate: P5, P10, P25(Q1), P50(Median), P75(Q3), P90, P95
    percentiles_to_calc = [5, 10, 25, 50, 75, 90, 95]
    calculated_percentiles = np.percentile(data_list, percentiles_to_calc)
    
    p5, p10, q1, median, q3, p90, p95 = calculated_percentiles
    iqr = q3 - q1
    
    return p5, p10, q1, median, q3, p90, p95, iqr, count

def main():
    parser = argparse.ArgumentParser(
        description="Calculate median, IQR, and other percentiles for experiment results for each recorded step, with method filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory containing the log files (e.g., *_distances.txt)."
    )
    parser.add_argument(
        "--print_steps",
        type=str,
        default="0,999,1999,2499,2999,3999", # Default from gradient_flow_digits.py
        help="Comma-separated string of the actual training step numbers that were logged. "
             "Used for labeling the output. Example: '0,19,39,59,79,99'"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None, 
        help="Comma-separated list of loss_types to consider (e.g., 'twd,fw_twd,fw_twd_rp'). If None, all found methods are processed."
    )
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    if not log_path.is_dir():
        print(f"Error: Log directory '{args.log_dir}' not found or is not a directory.")
        return

    target_methods_list = None
    if args.methods:
        target_methods_list = [m.strip() for m in args.methods.split(',') if m.strip()]
        if not target_methods_list: # Handles if --methods="" or --methods=" , "
            print("Warning: --methods argument provided but resulted in an empty list. Processing all methods.")
        else:
            print(f"INFO: Targeting methods: {', '.join(target_methods_list)}")
    else:
        print("INFO: No --methods specified, processing all found methods.")

    print_steps_list = None # Default to None if parsing fails or not provided
    if args.print_steps:
        try:
            # Filter out empty strings that might result from "1,,2"
            parsed_steps = [s.strip() for s in args.print_steps.split(',') if s.strip()]
            if not parsed_steps: # If args.print_steps was just commas or empty
                 raise ValueError("Parsed print_steps list is empty.")
            print_steps_list = [int(s) for s in parsed_steps]
        except ValueError:
            print(f"Error: Invalid format for --print_steps. Please use comma-separated integers. Got: '{args.print_steps}'")
            print("Will use 0-based index for steps instead of actual step numbers for labeling.")
            print_steps_list = None # Ensure it's None on failure

    aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    processed_files_count = 0
    found_files_count = 0

    for filepath in log_path.glob("images_*_distances.txt"):
        found_files_count += 1
        parsed_info = parse_filename(filepath.name)
        
        if parsed_info:
            loss_type, lr_val, _ = parsed_info # lr_val here should be a float
            if lr_val != 1e-3:
                continue
            
            # Apply method filter
            if target_methods_list and loss_type not in target_methods_list:
                continue
            
            try:
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if lines:
                    processed_files_count +=1 
                    for step_idx, line_content in enumerate(lines):
                        try:
                            distance = float(line_content)
                            aggregated_data[loss_type][lr_val][step_idx].append(distance)
                        except ValueError:
                            print(f"Warning: Could not parse distance '{line_content}' at step index {step_idx} in {filepath.name}")
                else:
                    print(f"Warning: Empty or effectively empty distances file (no valid lines): {filepath.name}")
            except Exception as e:
                print(f"Error processing file {filepath.name}: {e}")
        # else: # parse_filename handles its own warnings for non-matching files
            # pass

    if found_files_count == 0:
        print(f"No 'images_*_distances.txt' files found in '{args.log_dir}'.")
        return
    if processed_files_count == 0:
        if target_methods_list:
            print(f"No files processed for the target methods: {', '.join(target_methods_list)}. Check filenames, method names, or if log files are empty.")
        else:
            print(f"No files were processed. Check if log files are empty or if there are parsing issues.")
        return
    if not aggregated_data: 
        print(f"Data was processed from {processed_files_count} files, but no data could be aggregated into the final structure. Please check warnings above.")
        return

    print(f"\n--- Statistics for Distances per Step (from {processed_files_count} successfully processed files with data) ---")
    # 4 info columns + 7 percentiles + 1 IQR = 12 columns
    num_stat_cols = 8 
    num_info_cols = 4
    total_cols = num_info_cols + num_stat_cols

    header_format = "{:<20} {:<10} {:<12} {:<10} " + "{:<10} " * num_stat_cols 
    table_header = header_format.format(
        'Loss Type', 'LR', 'Actual Step', 'N Seeds', 
        'P5', 'P10', 'Q1(P25)', 'Median', 'Q3(P75)', 'P90', 'P95', 'IQR'
    )
    print(table_header)
    print("-" * len(table_header)) # Dynamic width based on actual header length

    all_loss_lr_pairs = []
    for loss_type_key in aggregated_data:
        for lr_val_key in aggregated_data[loss_type_key]:
            all_loss_lr_pairs.append((loss_type_key, lr_val_key))

    if not all_loss_lr_pairs:
        print("No (loss_type, lr_val) pairs were aggregated from the data. This shouldn't happen if aggregated_data was populated.")
        return

    sorted_loss_lr_keys = sorted(all_loss_lr_pairs, key=lambda x: (x[0], x[1]))

    for loss_type, lr_val in sorted_loss_lr_keys: 
        data_for_loss_lr = aggregated_data[loss_type][lr_val]
        # This check is a safeguard; if (loss_type, lr_val) is in sorted_loss_lr_keys, data_for_loss_lr should exist.
        if not data_for_loss_lr: 
            continue 
        
        max_step_idx_for_group = 0
        if data_for_loss_lr.keys(): 
            max_step_idx_for_group = max(data_for_loss_lr.keys())
        # else: # If a (loss_type, lr_val) somehow has no steps, skip (already covered by outer continue)
            # print(f"No step data found for Loss: {loss_type}, LR: {lr_val:g} despite key existing.")
            # continue

        first_line_for_group = True
        for step_idx in range(max_step_idx_for_group + 1):
            distances_at_step = data_for_loss_lr.get(step_idx, []) 
            
            # p5, p10, q1, median, q3, p90, p95, iqr, count
            stats_results = calculate_statistics(distances_at_step)
            count = stats_results[-1] # Last element is count
            
            actual_step_label = f"idx_{step_idx}" 
            if print_steps_list: # Check if print_steps_list was successfully parsed
                if step_idx < len(print_steps_list):
                    actual_step_label = str(print_steps_list[step_idx])
                else:
                    actual_step_label = f"idx_{step_idx}?" # step_idx is out of range of known print_steps
            
            loss_label_to_print = loss_type if first_line_for_group else ""
            lr_label_to_print = f"{lr_val:g}" if first_line_for_group else ""

            if count > 0:
                # Format numerical stats; stats_results[:-1] are the p5...iqr values
                formatted_stats = [f"{s:.3e}" for s in stats_results[:-1]]
                print(header_format.format(
                    loss_label_to_print,
                    lr_label_to_print,
                    actual_step_label,
                    count,
                    *formatted_stats
                ))
                first_line_for_group = False
            elif first_line_for_group : 
                print(header_format.format(
                    loss_label_to_print,
                    lr_label_to_print,
                    actual_step_label,
                    0, 
                    *["N/A"] * num_stat_cols # P5 to IQR are N/A
                ))
                first_line_for_group = False 
        
        if not first_line_for_group : 
            print(header_format.format(*[""] * total_cols)) # Print a blank separator line

if __name__ == "__main__":
    main()
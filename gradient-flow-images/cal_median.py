#!/usr/bin/env python
"""
Identifies seeds that produce final distances around the median for specified methods,
and lists or copies their corresponding images from a chosen step.
"""
import argparse
from pathlib import Path
import re
import numpy as np
from collections import defaultdict
import shutil # For copying files

def parse_filename(filename_str: str, file_suffix: str = "_distances.txt"):
    """
    Parses a filename to extract loss_type, lr, and seed.
    Example filename: images_sw_lr0.001_seed1_distances.txt
    """
    # Ensure file_suffix is escaped for regex if it contains special characters
    # For "_distances.txt", escaping isn't strictly needed but it's good practice.
    escaped_suffix = re.escape(file_suffix)
    # Regex: images_{loss_type}_lr{lr_value}_seed{seed_value}{suffix}
    pattern = re.compile(rf"images_(.+)_lr([0-9.eE+-]+)_seed(\d+){escaped_suffix}")
    match = pattern.match(Path(filename_str).name)
    
    if match:
        loss_type = match.group(1)
        lr_str = match.group(2)
        seed_str = match.group(3)
        try:
            lr_val = float(lr_str)
            seed_val = int(seed_str)
            return loss_type, lr_val, seed_val
        except ValueError:
            print(f"Warning: Could not parse numeric lr/seed from filename parts: '{lr_str}', '{seed_str}' in '{filename_str}'")
            return None
    else:
        # Useful for debugging if files are not matching as expected.
        # print(f"Debug: Filename '{filename_str}' did not match pattern for suffix '{file_suffix}'.")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Identify seeds that produce results around the median final distance and list/copy their images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing the log files (*_distances.txt).")
    parser.add_argument("--image_dir", type=str, default="imgs", help="Directory containing the generated images.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to copy selected images. If None, images are only listed. Directory will be created if it doesn't exist.")
    parser.add_argument("--methods", type=str, default="twd,fw_twd,fw_twd_rp", help="Comma-separated list of loss_types to consider.")
    parser.add_argument("--print_steps", type=str, default="0,999,1999,2499,2999,3999", help="Comma-separated string of the actual training step numbers that were logged (must match steps for which images were saved).")
    parser.add_argument("--image_step_index", type=int, default=-2, help="Index of the step in --print_steps list for which to get the image (e.g., -1 for last saved step, -2 for second to last saved step, 0 for first). Your '[-2]' would correspond to -2 here.")

    args = parser.parse_args()

    target_methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if not target_methods:
        print("Error: No target methods specified or all methods are empty strings. Please check the --methods argument.")
        return
    
    log_path = Path(args.log_dir)
    image_path = Path(args.image_dir)
    output_path = Path(args.output_dir) if args.output_dir else None

    if not log_path.is_dir():
        print(f"Error: Log directory '{log_path}' not found.")
        return
    if not image_path.is_dir():
        print(f"Error: Image directory '{image_path}' not found.")
        return
    
    if output_path:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create output directory '{output_path}': {e}")
            return

    print_steps_list = []
    if args.print_steps:
        try:
            print_steps_list = [int(s.strip()) for s in args.print_steps.split(',') if s.strip()]
            if not print_steps_list: # Check if list is empty after stripping and splitting
                raise ValueError("Parsed print_steps_list is empty.")
        except ValueError as e:
            print(f"Error: Invalid format or empty --print_steps: '{args.print_steps}'. Expected comma-separated non-empty integers. {e}")
            return
    else: # This case should ideally not be reached due to argparse default, but good for safety
        print("Error: --print_steps argument cannot be empty.")
        return

    image_step_num = -2
    try:
        # This ensures print_steps_list is not empty before indexing
        if not print_steps_list : raise IndexError("Cannot determine image step from empty print_steps_list.")
        image_step_num = print_steps_list[args.image_step_index]
    except IndexError:
        print(f"Error: --image_step_index {args.image_step_index} is out of bounds for the parsed print_steps list (length {len(print_steps_list)}).")
        print(f"Parsed print_steps: {print_steps_list}")
        print(f"Valid Python indices are from {-len(print_steps_list)} to {len(print_steps_list)-1}.")
        return
    
    print(f"INFO: Targeting methods: {', '.join(target_methods)}")
    print(f"INFO: Will select images from training step: {image_step_num} (derived from index {args.image_step_index} of print_steps {print_steps_list})")

    # Structure: { (loss_type, lr_val): [(final_distance, seed), ...], ... }
    final_distances_data = defaultdict(list)
    found_files_count = 0
    processed_files_count = 0 # Counts files that match pattern AND are for target methods

    for filepath in log_path.glob("images_*_distances.txt"):
        found_files_count += 1
        parsed_info = parse_filename(filepath.name, "_distances.txt")
        
        if parsed_info:
            loss_type, lr_val, seed = parsed_info
            
            if loss_type not in target_methods:
                continue # Skip if not one of the target methods

            if lr_val != 1e-3:
                continue
            
            processed_files_count +=1 
            try:
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()] # Read non-empty lines
                if lines:
                    final_distance = float(lines[-2]) # Assumes the last line is the final distance
                    final_distances_data[(loss_type, lr_val)].append((final_distance, seed))
                else:
                    print(f"Warning: Empty or effectively empty distances file: {filepath.name}")
            except ValueError as e: # Specifically for float conversion error
                print(f"Warning: Could not parse final distance in {filepath.name}: {e}")
            except Exception as e: # For other errors like file IO
                print(f"Error processing file {filepath.name}: {e}")

    if found_files_count == 0:
        print(f"No '..._distances.txt' files found in '{log_path}'.")
        return
    if processed_files_count == 0: # No files were processed for the target methods
        print(f"No distance files found or processed for the target methods: {', '.join(target_methods)}. Check filenames or --methods argument.")
        return
    if not final_distances_data: # No data was successfully aggregated
        print(f"No data collected for target methods. Ensure logs exist and filenames match expected patterns, and that distance files are not empty/corrupt.")
        return

    print("\n--- Seeds with Final Distances Closest to the Group Median ---")
    # Adjust column widths if filenames become too long
    header = f"{'Loss Type':<18} {'LR':<10} {'Med. Seed':<10} {'Seed Dist.':<12} {'Target Med.':<12} {'Sel. Step Img.':<75} {'GT Img.':<75}"
    print(header)
    print("-" * len(header))

    # Sort by loss_type (string) then by lr_val (float)
    sorted_keys = sorted(final_distances_data.keys(), key=lambda x: (x[0], float(x[1])))

    for loss_type, lr_val in sorted_keys:
        distance_seed_pairs = final_distances_data[(loss_type, lr_val)]
        # This check is mostly a safeguard, as defaultdict would create an empty list
        # but we only add to sorted_keys if final_distances_data has the key.
        if not distance_seed_pairs: 
            continue

        all_final_distances = [d[0] for d in distance_seed_pairs]
        target_median_distance = np.percentile(all_final_distances, 50)

        best_seed = -1
        best_seed_distance = float('inf') # Distance of the chosen best_seed
        
        # Find seed whose final distance is closest to target_median_distance
        # Store candidates as dictionaries to sort by multiple criteria for tie-breaking
        candidates = []
        for dist, seed_val in distance_seed_pairs:
            abs_diff = abs(dist - target_median_distance)
            candidates.append({'diff': abs_diff, 'seed': seed_val, 'dist': dist})
        
        # Sort candidates: primary key is absolute difference, secondary is seed number (lower seed wins ties)
        candidates.sort(key=lambda c: (c['diff'], c['seed']))
        
        if candidates: # If there are any candidates
            best_candidate = candidates[0]
            best_seed = best_candidate['seed']
            best_seed_distance = best_candidate['dist']
        
        if best_seed != -1:
            # lr_val is a float, :g format provides flexible float formatting (e.g., 0.001 or 1e-3)
            image_tag_basename = f"images_{loss_type}_lr{lr_val:g}_seed{best_seed}"
            
            generated_image_name = f"{image_tag_basename}_step{image_step_num:04d}.png"
            gt_image_name = f"{image_tag_basename}_GT.png"

            full_gen_img_path = image_path / generated_image_name
            full_gt_img_path = image_path / gt_image_name
            
            # Check for file existence and provide status
            gen_img_status = " (exists)" if full_gen_img_path.exists() else " (MISSING!)"
            gt_img_status = " (exists)" if full_gt_img_path.exists() else " (MISSING!)"
            
            # Prepare display names (can be truncated if too long, but for now, full name)
            display_gen_img_name = generated_image_name 
            display_gt_img_name = gt_image_name
            
            print(f"{loss_type:<18} {lr_val:<10g} {best_seed:<10} {best_seed_distance:<12.4e} {target_median_distance:<12.4e} {display_gen_img_name + gen_img_status:<75} {display_gt_img_name + gt_img_status:<75}")

            if output_path:
                if full_gen_img_path.exists():
                    try:
                        shutil.copy(full_gen_img_path, output_path / generated_image_name)
                    except Exception as e:
                        print(f"Error copying {full_gen_img_path} to {output_path}: {e}")
                # Only warn about missing if it's not just a filename but a path that was expected
                elif not Path(generated_image_name).is_file() and Path(generated_image_name).parent != Path('.'):
                     print(f"Warning: Cannot copy. Source file missing: {full_gen_img_path}")
                
                if full_gt_img_path.exists():
                    try:
                        shutil.copy(full_gt_img_path, output_path / gt_image_name)
                    except Exception as e:
                        print(f"Error copying {full_gt_img_path} to {output_path}: {e}")
                elif not Path(gt_image_name).is_file() and Path(gt_image_name).parent != Path('.'):
                    print(f"Warning: Cannot copy. Source file missing: {full_gt_img_path}")
        else: # This case should ideally not be reached if distance_seed_pairs was populated
            print(f"{loss_type:<18} {lr_val:<10g} {'N/A':<10} {'N/A':<12} {target_median_distance:<12.4e} {'N/A':<75} {'N/A':<75}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import multiprocessing
import sys
import os

# --- Pattern Checking Logic (32x32 specific) ---
def count_pattern_matches(a, b):
    """
    Calculates sums of partial products (a[i]*b[j]) for a 32x32 multiplication
    and checks them against a predefined set of properties ("cases").
    """
    N = 32
    a_bits = [(a >> i) & 1 for i in range(N)]
    b_bits = [(b >> i) & 1 for i in range(N)]
    comb = {}
    for i in range(N):
        for j in range(N):
            comb[(i, j)] = a_bits[i] * b_bits[j]

    results = {}
    case_counter = 0
    for k in range(1, 2 * N - 2):  # k from 1 to 61 for N=32
        current_Sk_sum = 0
        i_start = max(0, k - (N - 1))
        i_end = min(k, N - 1)
        for i in range(i_start, i_end + 1):
            j = k - i
            if 0 <= j < N:
                current_Sk_sum += comb[(i, j)]

        if k < N: M_k = k + 1
        else: M_k = 2 * N - 1 - k
        is_N_minus_1_sum = (k == N - 1)

        evaluated_properties_for_Sk = []
        if M_k == 2:
            evaluated_properties_for_Sk.append(current_Sk_sum == 2)
        elif M_k == 3:
            evaluated_properties_for_Sk.append(current_Sk_sum >= 2)
        else:  # M_k >= 4
            temp_specific_props = []
            L_bound = 2
            while L_bound <= M_k:
                if is_N_minus_1_sum and L_bound == M_k: break
                if L_bound == M_k:
                    temp_specific_props.append(current_Sk_sum == M_k)
                    L_bound += 1 
                elif L_bound + 1 == M_k:
                    if is_N_minus_1_sum:
                        temp_specific_props.append(current_Sk_sum >= L_bound and current_Sk_sum < M_k)
                    else:
                        temp_specific_props.append(current_Sk_sum >= L_bound)
                    L_bound += 2 
                else:
                    temp_specific_props.append(current_Sk_sum >= L_bound and current_Sk_sum < (L_bound + 2))
                    L_bound += 2 
            evaluated_properties_for_Sk.extend(temp_specific_props)
            evaluated_properties_for_Sk.append(current_Sk_sum >= 2)

        for prop_result in evaluated_properties_for_Sk:
            case_counter += 1
            results[case_counter] = prop_result
    return results

# --- Case to Property Mapping Generation ---
def generate_case_property_mapping_32x32():
    """
    Generates a mapping of case numbers to their corresponding property descriptions
    for a 32x32 multiplier.
    """
    N = 32
    mapping = {}
    case_counter = 0

    for k in range(1, 2 * N - 2):  # k from 1 to 61
        sk_terms_list = []
        i_start = max(0, k - (N - 1))
        i_end = min(k, N - 1)
        for i in range(i_start, i_end + 1):
            j = k - i
            sk_terms_list.append(f"a[{i}]*b[{j}]")
        
        sk_sum_expression_string = " + ".join(sk_terms_list)

        if k < N: M_k = k + 1
        else: M_k = 2 * N - 1 - k
        is_N_minus_1_sum = (k == N - 1)

        property_description_strings_for_Sk = []
        if M_k == 2:
            property_description_strings_for_Sk.append(f"({sk_sum_expression_string}) == 2")
        elif M_k == 3:
            property_description_strings_for_Sk.append(f"({sk_sum_expression_string}) >= 2")
        else:  # M_k >= 4
            temp_specific_descs = []
            L_bound = 2
            while L_bound <= M_k:
                if is_N_minus_1_sum and L_bound == M_k: break
                desc = ""
                if L_bound == M_k:
                    desc = f"({sk_sum_expression_string}) == {M_k}"
                    L_bound += 1
                elif L_bound + 1 == M_k:
                    if is_N_minus_1_sum:
                        desc = f"({sk_sum_expression_string}) >= {L_bound} AND ({sk_sum_expression_string}) < {M_k}"
                    else:
                        desc = f"({sk_sum_expression_string}) >= {L_bound}"
                    L_bound += 2
                else:
                    desc = f"({sk_sum_expression_string}) >= {L_bound} AND ({sk_sum_expression_string}) < {L_bound + 2}"
                    L_bound += 2
                temp_specific_descs.append(desc)
            property_description_strings_for_Sk.extend(temp_specific_descs)
            property_description_strings_for_Sk.append(f"({sk_sum_expression_string}) >= 2")

        for desc_str in property_description_strings_for_Sk:
            case_counter += 1
            mapping[case_counter] = desc_str
    return mapping

# --- Worker function for multiprocessing ---
def worker_process_item(a_b_tuple):
    a_val, b_val = a_b_tuple
    return count_pattern_matches(a_val, b_val)

# --- Utility Functions ---
def convert_to_int(value):
    if isinstance(value, str):
        original_value = value
        if value.startswith('0b'): value = value[2:]
        elif value.startswith('b'): value = value[1:]
        try:
            return int(value, 2)
        except ValueError:
            try: return int(value)
            except ValueError:
                raise ValueError(f"Could not convert string to int: '{original_value}'")
    elif isinstance(value, (int, np.integer)):
        return int(value)
    else:
        raise ValueError(f"Unexpected value type: {type(value)}. Value: {value}")

# --- Modified Analysis Logic for Automation ---
def analyze_csv_for_automation(filename, total_generated_cases):
    """
    Analyzes a single CSV file for pattern matches.
    Results are saved in the same directory as the input CSV.
    """
    print(f"--- Analyzing: {filename} ---")
    df = pd.read_csv(filename, low_memory=False)
    
    op_a_col, op_b_col = None, None
    for col in df.columns:
        col_lower = col.lower()
        if 'op_a' in col_lower or ('operand' in col_lower and 'a' in col_lower) or col_lower == 'a': op_a_col = col
        if 'op_b' in col_lower or ('operand' in col_lower and 'b' in col_lower) or col_lower == 'b': op_b_col = col
    if op_a_col is None:
        if 'operand_0' in df.columns: op_a_col = 'operand_0'
        elif 'A' in df.columns: op_a_col = 'A'
    if op_b_col is None:
        if 'operand_1' in df.columns: op_b_col = 'operand_1'
        elif 'B' in df.columns: op_b_col = 'B'

    if op_a_col is None or op_b_col is None:
        print(f"Could not automatically identify op_a/op_b columns in {filename} from: {df.columns.tolist()}")
        # In an automated script, prompting might not be ideal. Consider logging and skipping.
        # For now, let's raise an error for this specific file.
        raise ValueError(f"Operand columns not found in {filename}")
    print(f"Using columns '{op_a_col}' (A) and '{op_b_col}' (B) for {filename}")

    df['op_a_int'] = df[op_a_col].apply(convert_to_int)
    df['op_b_int'] = df[op_b_col].apply(convert_to_int)
    
    case_counts = {i: 0 for i in range(1, total_generated_cases + 1)}
    
    operands_list = list(zip(df['op_a_int'], df['op_b_int']))
    total_operations = len(operands_list)
    
    if total_operations == 0:
        print(f"No operations to process in {filename}.")
        return

    print(f"Processing {total_operations} operations from {filename} using multiple cores...")
    
    num_processes = os.cpu_count() - 1 if os.cpu_count() is not None and os.cpu_count() > 1 else 1
    chunk_size = max(1, total_operations // (num_processes * 10)) if total_operations > (num_processes * 10) else 1
    if total_operations < 100 : chunk_size = 1
    print(f"Using {num_processes} worker processes with chunksize {chunk_size} for {filename}.")

    processed_count = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_process_item, operands_list, chunksize=chunk_size)
        for i, matches_dict in enumerate(results_iterator):
            for case_num, matched in matches_dict.items():
                if case_num in case_counts:
                    if matched: case_counts[case_num] += 1
                else:
                    print(f"Warning for {filename}: Unexpected case number {case_num} (max expected: {total_generated_cases}).")
            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_operations:
                print(f"Aggregated results for {processed_count}/{total_operations} ops from {filename}...")
    
    print(f"Finished processing {total_operations} operations from {filename}.")
    
    # Save results to the directory of the input CSV
    output_dir = os.path.dirname(filename)
    if not output_dir: # If filename is just 'file.csv', dirname is empty
        output_dir = "." 
    analysis_output_filename = os.path.join(output_dir, "pattern_analysis_results_32x32.txt")
    
    with open(analysis_output_filename, "w") as f:
        f.write(f"Analysis Results for {filename} (32x32 Multiplier Patterns)\n")
        f.write(f"Total operations: {total_operations}\n\n")
        f.write("Case Pattern Matches:\n")
        f.write("-" * 50 + "\n")
        for case_num in range(1, total_generated_cases + 1):
            count = case_counts.get(case_num, 0)
            percentage = (count / total_operations) * 100 if total_operations > 0 else 0
            f.write(f"Case {case_num:3d}: {count:7d} matches ({percentage:6.2f}%)\n")
    print(f"Analysis results for {filename} saved to {analysis_output_filename}")

# --- Automation Script Logic ---
def find_and_process_csvs(project_root_dir):
    print(f"Starting analysis in project root: {project_root_dir}")
    
    # 1. Generate the case-to-property mapping file ONCE in the project root
    mapping_filepath = os.path.join(project_root_dir, "case_property_mapping_32x32.txt")
    case_to_property_map = generate_case_property_mapping_32x32()
    
    if not case_to_property_map:
        print("Error: Case to property map could not be generated. Aborting automation.")
        return

    with open(mapping_filepath, "w") as f_map:
        f_map.write("Case Number to Property Mapping (N=32 Multiplier)\n")
        f_map.write("Each property refers to the sum of terms like a[i]*b[j].\n")
        f_map.write("The full sum expression is shown in parentheses for each property.\n")
        f_map.write("-" * 70 + "\n")
        for case_num in sorted(case_to_property_map.keys()):
            f_map.write(f"Case {case_num:3d}: {case_to_property_map[case_num]}\n")
    print(f"Global case-to-property mapping saved to {mapping_filepath}")

    TOTAL_GENERATED_CASES = max(case_to_property_map.keys()) if case_to_property_map else 0
    if TOTAL_GENERATED_CASES == 0:
        print("Warning: TOTAL_GENERATED_CASES is 0 from mapping. Analysis might be incomplete.")
        # Consider setting a default or raising an error if this is critical
        # For now, proceed, but case_counts might be empty or small.

    # 2. Walk through directories and find target CSV files
    target_csv_filename = "mult_ops_binary.csv"
    found_csv_files = []

    for dirpath, _, filenames in os.walk(project_root_dir):
        if target_csv_filename in filenames:
            full_csv_path = os.path.join(dirpath, target_csv_filename)
            found_csv_files.append(full_csv_path)
            print(f"Found '{target_csv_filename}' in: {dirpath}")

    if not found_csv_files:
        print(f"No '{target_csv_filename}' files found under '{project_root_dir}'.")
        return

    # 3. Process each found CSV
    print(f"\nFound {len(found_csv_files)} CSV file(s) to process.")
    for csv_file_path in found_csv_files:
        try:
            analyze_csv_for_automation(csv_file_path, TOTAL_GENERATED_CASES)
        except Exception as e:
            print(f"!!! Critical Error processing {csv_file_path}: {e}")
            print(f"!!! Skipping this file and continuing with others if any.")
            # Optionally: log this error to a separate error log file
    
    print("\nAutomation finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # This is important for multiprocessing on Windows, and good practice otherwise
    multiprocessing.freeze_support() 

    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "." 
        print(f"No project root directory provided. Using current directory: '{os.path.abspath(project_root)}'")

    if not os.path.isdir(project_root):
        print(f"Error: Provided project root '{project_root}' is not a valid directory.")
    else:
        find_and_process_csvs(os.path.abspath(project_root)) # Use absolute path
import pandas as pd
import numpy as np # For np.integer type check in convert_to_int
import multiprocessing
import sys

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
    for k in range(1, 2 * N - 2): # k from 1 to 61
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
        else: # M_k >= 4
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
        
        # sk_sum_expression_string will be like "a[0]*b[1] + a[1]*b[0]"
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
            property_description_strings_for_Sk.append(f"({sk_sum_expression_string}) >= 2") # General case

        for desc_str in property_description_strings_for_Sk:
            case_counter += 1
            mapping[case_counter] = desc_str
            
    return mapping

# --- Worker function for multiprocessing ---
# This function must be defined at the top level for pickling by multiprocessing
def worker_process_item(a_b_tuple):
    """
    Wrapper function for multiprocessing. Calls count_pattern_matches.
    """
    a_val, b_val = a_b_tuple
    return count_pattern_matches(a_val, b_val)

# --- Utility Functions ---
def convert_to_int(value):
    if isinstance(value, str):
        original_value = value # Keep original for error message if all parsing fails
        if value.startswith('0b'): value = value[2:]
        elif value.startswith('b'): value = value[1:]
        try:
            return int(value, 2)
        except ValueError:
            try: # Fallback: try to interpret as a decimal string
                return int(value)
            except ValueError:
                raise ValueError(f"Could not convert string to int: '{original_value}'")
    elif isinstance(value, (int, np.integer)): # Handle numpy integers as well
        return int(value)
    else:
        raise ValueError(f"Unexpected value type: {type(value)}. Value: {value}")

# --- Main Analysis Logic ---
def analyze_csv(filename):
    # Generate and save the case-to-property mapping first
    mapping_filename = "case_property_mapping_32x32.txt"
    case_to_property_map = generate_case_property_mapping_32x32()
    # with open(mapping_filename, "w") as f_map:
    #     f_map.write(f"Case Number to Property Mapping (N=32 Multiplier)\n")
    #     f_map.write("Each S_k property refers to the sum of a[i]*b[j] terms where i+j=k.\n")
    #     f_map.write("The expression for S_k is shown in parentheses for each property.\n")
    #     f_map.write("-" * 70 + "\n")
    #     for case_num in sorted(case_to_property_map.keys()):
    #         f_map.write(f"Case {case_num:3d}: {case_to_property_map[case_num]}\n")
    # print(f"Case to property mapping saved to {mapping_filename}")

    TOTAL_GENERATED_CASES = 0
    if case_to_property_map:
        TOTAL_GENERATED_CASES = max(case_to_property_map.keys())
    if TOTAL_GENERATED_CASES == 0:
        print("Warning: No cases were generated for property mapping. Analysis might be incorrect.")
        # Fallback to a known number if needed, but dynamic is better
        # TOTAL_GENERATED_CASES = 552 # Previous hardcoded estimate

    df = pd.read_csv(filename, low_memory=False)
    print(f"\nColumns in CSV: {df.columns.tolist()}")
    
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
        print(f"Could not automatically identify op_a and op_b columns from: {df.columns.tolist()}")
        op_a_col = input("Please enter the column name for operand A: ")
        op_b_col = input("Please enter the column name for operand B: ")
        if op_a_col not in df.columns or op_b_col not in df.columns:
            raise ValueError("Specified operand columns not found in CSV.")
    print(f"Using column '{op_a_col}' for operand A and '{op_b_col}' for operand B.")

    df['op_a_int'] = df[op_a_col].apply(convert_to_int)
    df['op_b_int'] = df[op_b_col].apply(convert_to_int)
    
    case_counts = {i: 0 for i in range(1, TOTAL_GENERATED_CASES + 1)}
    
    operands_list = list(zip(df['op_a_int'], df['op_b_int']))
    total_operations = len(operands_list)
    
    print(f"\nProcessing {total_operations} operations using multiple cores...")
    
    num_processes = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
    # Adjust chunksize based on data: rule of thumb, ensure each worker gets a few chunks
    chunk_size = max(1, total_operations // (num_processes * 10)) if total_operations > (num_processes * 10) else 1
    if total_operations < 100 : chunk_size = 1 # Small datasets process one by one per worker call

    print(f"Using {num_processes} worker processes with chunksize {chunk_size}.")

    processed_count = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_process_item, operands_list, chunksize=chunk_size)
        
        for i, matches_dict in enumerate(results_iterator):
            for case_num, matched in matches_dict.items():
                if case_num in case_counts:
                    if matched: case_counts[case_num] += 1
                else:
                    print(f"Warning: Unexpected case number {case_num} from worker (max expected: {TOTAL_GENERATED_CASES}).")
            
            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_operations:
                print(f"Aggregated results for {processed_count}/{total_operations} operations...")
    
    print(f"Finished processing all {total_operations} operations.")
    
    print(f"\nAnalysis Results for {filename}")
    print(f"Total operations: {total_operations}")
    print("\nCase Pattern Matches:")
    print("-" * 50)
    
    for case_num in range(1, TOTAL_GENERATED_CASES + 1):
        count = case_counts.get(case_num, 0)
        percentage = (count / total_operations) * 100 if total_operations > 0 else 0
        print(f"Case {case_num:3d}: {count:7d} matches ({percentage:6.2f}%)")
    
    output_filename = "pattern_analysis_results_32x32.txt"
    with open(output_filename, "w") as f:
        f.write(f"Analysis Results for {filename} (32x32 Multiplier Patterns)\n")
        f.write(f"Total operations: {total_operations}\n\n")
        f.write("Case Pattern Matches:\n")
        f.write("-" * 50 + "\n")
        for case_num in range(1, TOTAL_GENERATED_CASES + 1):
            count = case_counts.get(case_num, 0)
            percentage = (count / total_operations) * 100 if total_operations > 0 else 0
            f.write(f"Case {case_num:3d}: {count:7d} matches ({percentage:6.2f}%)\n")
    print(f"\nResults saved to {output_filename}")

# --- Script Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        csv_filename = "mult_ops_binary.csv" 
        print(f"No CSV filename provided, using default: '{csv_filename}'")
    
    try:
        analyze_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Troubleshooting: Ensure CSV is correctly formatted and operand columns are valid.")
        try:
            df_debug = pd.read_csv(csv_filename, nrows=5)
            print(f"\nFirst 5 rows of '{csv_filename}' for inspection:")
            print(df_debug)
        except Exception as e2:
            print(f"Could not read the CSV file for debugging: {e2}")
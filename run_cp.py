import numpy as np
import os
import torch


import statistics
import math
import argparse



# CP

def calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=False):

    s_t = torch.abs(target_calib-mu_calib) / sd_calib
    if gc:
        S = (s_t).mean().sqrt()
        if alpha == 0.1:
            q = 1.64485 * S.item()
        elif alpha == 0.05:
            q = 1.95996 * S.item()
        else:
            print("Choose another value of alpha!! (0.1 / 0.05)")
    else:
        s_t_sorted, _ = torch.sort(s_t, dim=0)
        q_index = math.ceil((len(s_t_sorted)) * (1 - alpha))
        q = s_t_sorted[q_index].item()   
    return q

# CP/GC prediction

def calc_stats(q, target, mu, sd):
    lower = mu - q * sd
    upper = mu + q * sd
    length = torch.mean(abs(upper - lower))
    coverage = avg_cov(lower, upper, target)
    return length, coverage

def avg_cov(lower, upper, target):
    in_the_range = torch.sum((target  >= lower) & (target  <= upper)).item()
    coverage = in_the_range / len(target) * 100
    return coverage


def shuffle_arrays(calib_arrays, test_arrays):
    """
    Shuffles calibration and test arrays together, maintaining correspondence across arrays.

    Args:
        calib_arrays (list of tensors): List of calibration arrays to shuffle.
        test_arrays (list of tensors): List of test arrays to shuffle.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        tuple: Shuffled calibration arrays, shuffled test arrays.
    """
    # if seed is not None:
    #     np.random.seed(seed)

    # Combine calib and test arrays
    combined_arrays = [torch.cat([calib, test], dim=0) for calib, test in zip(calib_arrays, test_arrays)]
    
    # Generate shuffle indices
    total_length = combined_arrays[0].shape[0]
    shuffle_indices = np.random.permutation(total_length)

    # Apply shuffle indices
    shuffled_arrays = [arr[shuffle_indices] for arr in combined_arrays]

    # Split back into calib and test arrays
    split_index = len(calib_arrays[0])
    calib_shuffled = [arr[:split_index] for arr in shuffled_arrays]
    test_shuffled = [arr[split_index:] for arr in shuffled_arrays]

    return calib_shuffled, test_shuffled
   
    

def load_data(dataset_name, set_type):
    data_dir = './data'
    logvars = torch.tensor(np.load(f'{data_dir}/{dataset_name}/{set_type}/logvars.npy'))
    y_p = torch.tensor(np.load(f'{data_dir}/{dataset_name}/{set_type}/mu.npy'))
    targets = torch.tensor(np.load(f'{data_dir}/{dataset_name}/{set_type}/targets.npy'))
    return y_p, logvars, targets
    
    
    

def main(dataset, alpha, iters):
    
    # Load data  
    y_p_calib_original, logvars_calib_original, targets_calib_original = load_data(dataset, 'validation')
    y_p_test_original, logvars_test_original, targets_test_original = load_data(dataset, 'test')
    
    # Calibration and test arrays (from your original code)
    calib_arrays = [
        y_p_calib_original, 
        logvars_calib_original, 
        targets_calib_original, 
    ]

    test_arrays = [
        y_p_test_original, 
        logvars_test_original, 
        targets_test_original, 
    ]

    q_all = []
    avg_len_all = []
    avg_cov_all = []
    q_all_gc = []
    avg_len_all_gc = []
    avg_cov_all_gc = []
    
    avg_len_valid_all = []
    avg_cov_valid_all = []
    avg_len_valid_all_gc = []
    avg_cov_valid_all_gc = []
        

    for j in range(iters):
        print(f'Iter: {j}')
        y_p_calib = []
        logvars_calib = []
        targets_calib = []
        
        
        calib_shuffled, test_shuffled = shuffle_arrays(calib_arrays, test_arrays)
        y_p_calib, logvars_calib, targets_calib = calib_shuffled
        y_p_test, logvars_test, targets_test = test_shuffled
        
                    
        # validation set   
        y_p_calib = y_p_calib.clamp(0, 1).unsqueeze(1)
        mu_calib = y_p_calib.mean(dim=1)
        logvar_calib = logvars_calib.mean(dim=1).unsqueeze(1)
        var_calib  = logvar_calib.exp()
        sd_calib = var_calib.sqrt()
        target_calib = targets_calib.unsqueeze(1)




        
        # test set
                                 
        y_p_test = y_p_test.clamp(0, 1).unsqueeze(1)
        mu_test = y_p_test.mean(dim=1)
        logvar_test = logvars_test.mean(dim=1).unsqueeze(1)
        var_test  = logvar_test.exp()
        sd_test = var_test.sqrt()
        target_test = targets_test.unsqueeze(1)
            
        q = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha)
        
        valid_length, valid_coverage = calc_stats(q, target_calib, mu_calib, sd_calib)
        test_length, test_coverage = calc_stats(q, target_test, mu_test, sd_test)
                     
            
        q_gc = calc_optimal_q(target_calib, mu_calib, sd_calib, alpha, gc=True)
                     
        valid_length_gc, valid_coverage_gc = calc_stats(q_gc, target_calib, mu_calib, sd_calib)
        test_length_gc, test_coverage_gc = calc_stats(q_gc, target_test, mu_test, sd_test)
        
            
        print(f'q: {q}, q_gc: {q_gc}')
        print(f'valid_length: {valid_length}, valid_coverage: {valid_coverage}')
        print(f'test_length: {test_length}, test_coverage: {test_coverage}')
        print(f'valid_length_gc: {valid_length_gc}, valid_coverage_gc: {valid_coverage_gc}')
        print(f'test_length_gc: {test_length_gc}, test_coverage_gc: {test_coverage_gc}')

            

        q_all.append(q)
        avg_len_all.append(test_length.item())
        avg_cov_all.append(test_coverage)
        
        q_all_gc.append(q_gc)
        avg_len_all_gc.append(test_length_gc.item())
        avg_cov_all_gc.append(test_coverage_gc)
        
        avg_len_valid_all.append(valid_length.item())
        avg_cov_valid_all.append(valid_coverage)
        avg_len_valid_all_gc.append(valid_length_gc.item())
        avg_cov_valid_all_gc.append(valid_coverage_gc)
        
        
    print(f"q cp: {q_all}")
    print(f"q gc: {q_all_gc}")
    print(f"avg_len cp: {avg_len_all}")
    print(f"avg_len gc: {avg_len_all_gc}")
    print(f"avg_cov cp: {avg_cov_all}")
    print(f"avg_cov gc: {avg_cov_all_gc}")



    # Define the output file path
    output_dir= './results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"dataset_{dataset}_alpha_{alpha}_iterations_{iters}.txt"

    # Open the file in append mode
    with open(f'{output_dir}/{output_file}', "w") as f:
        # Print and save CP metrics
        print(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}')
        f.write(f'q CP mean: {statistics.mean(q_all)}, q CP std: {statistics.stdev(q_all)}\n')
        
        print(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}')
        f.write(f'avg_len CP mean: {statistics.mean(avg_len_all)}, avg_len CP std: {statistics.stdev(avg_len_all)}\n')
        
        print(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}')
        f.write(f'avg_cov CP mean: {statistics.mean(avg_cov_all)}, avg_cov CP std: {statistics.stdev(avg_cov_all)}\n')
        
        # Print and save GC metrics
        print(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}')
        f.write(f'q GC mean: {statistics.mean(q_all_gc)}, q GC std: {statistics.stdev(q_all_gc)}\n')
        
        print(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}')
        f.write(f'avg_len GC mean: {statistics.mean(avg_len_all_gc)}, avg_len GC std: {statistics.stdev(avg_len_all_gc)}\n')
        
        print(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}')
        f.write(f'avg_cov GC mean: {statistics.mean(avg_cov_all_gc)}, avg_cov GC std: {statistics.stdev(avg_cov_all_gc)}\n')
        

        # save validation results
        f.write(f"Validation results:\n")
        print(f"avg_len validation mean: {statistics.mean(avg_len_valid_all)}, avg_len validation std: {statistics.stdev(avg_len_valid_all)}")
        f.write(f"avg_len validation mean: {statistics.mean(avg_len_valid_all)}, avg_len validation std: {statistics.stdev(avg_len_valid_all)}\n")
        
        print(f"avg_cov validation mean: {statistics.mean(avg_cov_valid_all)}, avg_cov validation std: {statistics.stdev(avg_cov_valid_all)}")
        f.write(f"avg_cov validation mean: {statistics.mean(avg_cov_valid_all)}, avg_cov validation std: {statistics.stdev(avg_cov_valid_all)}\n")
        
        print(f"avg_len validation mean GC: {statistics.mean(avg_len_valid_all_gc)}, avg_len validation std GC: {statistics.stdev(avg_len_valid_all_gc)}")
        f.write(f"avg_len validation mean GC: {statistics.mean(avg_len_valid_all_gc)}, avg_len validation std GC: {statistics.stdev(avg_len_valid_all_gc)}\n")
        
        print(f"avg_cov validation mean GC: {statistics.mean(avg_cov_valid_all_gc)}, avg_cov validation std GC: {statistics.stdev(avg_cov_valid_all_gc)}")
        f.write(f"avg_cov validation mean GC: {statistics.mean(avg_cov_valid_all_gc)}, avg_cov validation std GC: {statistics.stdev(avg_cov_valid_all_gc)}\n")        
        

        # Print and save additional info
        print(f"{dataset}, {alpha}")
        f.write(f"{dataset}, {alpha}\n")
    
 
    


    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Calibrated Instance Dependent.")
    
    # Add arguments
    parser.add_argument("--dataset", type=str, required=True, help="data set name (name of the folder in the data directory).")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value")
    parser.add_argument("--iters", type=int, required=True, help="Number of iterations.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(dataset=args.dataset, alpha=args.alpha, iters=args.iters)
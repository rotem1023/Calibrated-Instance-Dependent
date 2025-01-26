## CP-VS and Gaussian-VS Implementation

This repo is an implementation CP-VS and  Gaussian-VS.

### Installation

Our experiments used the following versions:
 -  Python (11)
 -  torch (2.5.1)
 -  numpy (2.2.2)

### Directory Structure

To run the methods on a specific dataset, create the following directory structure under the ./data folder:

- {dataset name}
    - test
        - logvars.npy
        - mu.npy
        - targets.npy
    - validation
        - logvars.npy
        - mu.npy
        - targets.npy

#### File Descriptions:
    - logvars.npy - model predictions for log variance values.
    - mu.npy - model predictions for expectations values.
    - targets.npy - ground truth values.

### Running the Code

To execute the code, use the following command:

```
python run_cp.py  --dataset {dataset name} --alpha {alpha} --iters {number of iterations}
```

Results will be saved in the ./results folder.

note: currently the code support only alpha = 0.05 or alpha = 0.1

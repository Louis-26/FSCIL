# modification
## hyperparameter tuning
in this branch, I tune the hyperparameters, and rename the save path to distinguish the change
specifically, in the checkpoint folder, I will name each folder as `diffusion_fscil_cifar100_<diff_steps>_<ddim_steps>_<lr_diffusion>`

### first stage hyperparameter tuning
I modify the following hyperparameters:
- num_diffusion_steps: 1000, changed into one from [50,200,500]
- ddim_steps: 50, changed into one from [10, 30, 100]
- lr_diffusion: 1e-4, changed into one from [2e-5, 5e-5, 2e-4]

we will get in total 9 combinations(change one, get others fixed)

### first stage discovery
1. performance gets worse when learning rate is smaller

2. when diffusion steps is smaller, performance seems to be better

3. when ddim steps is smaller, performance seems to be better




# T-GRAB ☕ : A Synthetic Reasoning Benchmark for Learning On Temporal Graphs
This repository contains the implementation of the T-GRAB framework, a comprehensive set of synthetic tasks specifically designed to systematically evaluate the reasoning capabilities of Temporal Graph Neural Networks (TGNNs) over time. The framework offers controlled and interpretable tasks that target three core temporal reasoning skills:
a) Counting and memorizing periodic repetitions (periodicity),
b) Inferring delayed causal effects (cause_effect), and
c) Capturing long-range dependencies across both spatial and temporal dimensions (long_range).

The code can be used to reproduce the results from the original paper.

## Reproduce the results
### Installation
To get started with T-GRAB, follow these installation steps:

1. Navigate to the T-GRAB directory
2. Create and activate a Python virtual environment
3. Install the required dependencies

```bash
cd .../T-GRAB
python -m venv tgrab
pip install -r requirements.txt 
```
### Dataset generation
There are two ways to generate datasets in T-GRAB:

1. **Quick Generation**
   - Use the `sample.sh` scripts located in `scripts/dataset/`
   - This is suitable for quick testing and small datasets

2. **Slurm-based Generation**
   - Use `all_in_one.sh` scripts for distributed dataset creation
   - Configure dataset parameters in the script before running

   ```bash
   # For periodicity tasks
   ./scripts/dataset/periodicity/all_in_one.sh [sbm] [fixed_er]
   # sbm: stochastic periodicity
   # fixed_er: deterministic periodicity
   # You can run either one or both arguments together

   # For cause-effect tasks
   ./scripts/dataset/cause_effect/all_in_one.sh

   # For long-range dependency tasks
   ./scripts/dataset/long_range/all_in_one.sh
   ```

The generated datasets are:
- Stored in `scratch/data/` directory by default
- Saved in numpy compressed format for training
- Available in CSV format on [Hugging Face](https://huggingface.co/datasets/Gilestel/T-GRAB)

### Training
1. Make sure to have a separate directory for long-range
2. rename memory_node as cause_effect
2.5 explain the wandb configuration: login first and set the wandb entity
3. run all the methods only on one dataset using slurm
4. describe the quick version to only run on each method
5. explain that there could be two configs, one run in slurm and the other in local
6. explain which variables they need to set before running, after changing the arguments for environment and the dataset.
7. remove CTDG_do_snapshot training
    a. long_range
    b. cause_effect
    c. periodicity
8. describe there are two sets of metrics that they need to work: running-specific, dataset-specific, and model-specific variables.
9. Describe where the model weights are stored.
10. Give an example using CTAN.


## Contribution
Adding new methods
Adding new datasets
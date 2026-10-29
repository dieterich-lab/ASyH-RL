# ASyH-RL

The data synthesis tool based on principles of Q-learning for h-parameters optimization


## Overview

The fork of ASyH, enhanced with a module that runs a reinforcement learning optimization
of generative model hyperparameters. 
The RL-agent runs a closed optimization loop for predefined number of episodes, 
adjusting model parameters at the end of each iteration. 
Its main stages follow the order:
generation -> evaluation -> adjustment of parameters . 

At this moment there are 5 available models in the toolkit, although we plan to extend the list:

| ID | Model | Description |
|---:|---|---|
| 0 | CTGAN | Conditional tabular generative adversarial network |
| 1 | CopulaGAN | Copula generative adversarial network |
| 2 | TVAE | Tabular variational autoencoder |
| 3 | Gaussian Copula | — |
| 4 | Gaussian Mixture | — |


## Required dependencies & libraries

Necessary libraries are specified in the file `requirements.txt` . 
To install needed dependencies run in cmd inside the right virtual environment: 
`$ pip install -r requirements.txt`


## How to use

To launch the tool you can use the sbatch script `launch_sbatch_model.sh` - on HPC it submits a job to SLURM, 
that runs an RL-optimization routine for a model, whose number is provided as an argument; 
for example `$ sbatch launch_sbatch_model.sh 0` - it will start a process for CTGAN & RL agent, 
depending on what model you need replace the digit with another one based on the table above in the overview section.


## Essential files and folders

* `launch_sbatch_model.sh` - on HPC it submits a job to SLURM, that runs an RL-optimization routine for a model, whose number is 
provided as an argument; for example
`$ sbatch launch_sbatch_model.sh 0` - it will start a process for CTGAN & RL agent

* `main_rl_optimization.py` - the main python script

* `rl_agent/` - the folder contains the module with the routines for RL agent

* `evaluator/utility_evaluator_2.py` - responsible for calculating utility score of produced data

* `evaluator/fidelity_evaluator.py` - for calculating fidelity of synth. data based on statistical tests (KS, JS, TVD)

* `rl_optimizer.py` - the implementation of closed optimization loop


## About ASyH (the original toolkit)

A data protection tool that utilizes generative machine learning models to
create synthetic datasets, safeguarding sensitive patient information. The ASyH
is a software helping Clinics as holders of large quantities of highly
restricted personal health data to provide the Medical Data Community with
realistic datasets without the breach of privacy. It does this by synthesizing
data with Machine Learning techniques which preserve data distribution and
correlation while adding as much variation to the synthetic data, in order for
it to have no resemblance to any of the original patient data entries.
For synthesis, metrics and quality assurance we will mainly use the Synthetic Data Vault (github).

More info can be found here - https://github.com/dieterich-lab/ASyH/blob/main/README.md.


## References

1. Johann, T. I., Otte, K., Prasser, F., & Dieterich, C. (2025). Anonymize or synthesize? Privacy-preserving methods for heart failure score analytics. European Heart Journal-Digital Health, 6(1), 147-154.
2. Giomi, M., Boenisch, F., Wehmeyer, C., & Tasnádi, B. (2022). A unified framework for quantifying privacy risk in synthetic data. arXiv preprint arXiv:2211.10459.
3. Wang, S., Wang, J., Cheng, X., Zhang, W., Tian, Y., Ning, M., ... & Liu, Y. (2026). CLEO closed loop framework for synthesizing medical privacy preserving tabular data. npj Digital Medicine.
4. Montanez, A. (2018). SDV: an open source library for synthetic data generation (Doctoral dissertation, Massachusetts Institute of Technology).
# GraphDynamics
The goal of the project is to learn how genes interact with each other during cell development. We model genes as nodes in a static graph, and use graph message passing to predict next cell states.

## Training Data Prep
For now, we are working with 2 datasets: 1) EB data 2)worm body muscle data.

For preprocessing, we first clean the raw [Cell x Gene] data matrix, run MIOFlow to produce cellular trajectory in PCA dimension, and reverse PCA to get cellular trajectory in the ambient gene space. 

### Scripts to produce training data
1. EB data
* '/MIOFlow/notebooks/[graphs]EB-prep.ipynb'
* '/MIOFlow/notebooks/[graphs]EB-traj-gene.ipynb'

2. Worm body muscle data
* '/MIOFlow/notebooks/[Graphs] Worm-Body-traj-gene.ipynb'

## Architecture
1. Discrete (Residule/RNN)

2. Continuous (Graph ODE)

## How to run training script
```
sbatch batch_script.sh
```
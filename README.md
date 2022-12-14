# GraphDynamics
We propose a method to learn gene regulatory network that can predict high dimensional data trajectories from sporadic single-cell measurement data. Specifically in this project, we focus on learning a graph network with genes as the vertices and modeling regulatory interactions between the vertices to aligns with given single-cell data.

## Training Data Prep
For now, we are working with 2 datasets: 1) EB data and 2) C.elegans body wall muscle data.

For preprocessing, we first clean the raw [Cell x Gene] data matrix, run MIOFlow to produce cellular trajectory in PCA dimension, and reverse PCA to get cellular trajectory in the ambient gene space. 

### Scripts to produce training data
1. EB data
* '/MIOFlow/notebooks/[graphs]EB-prep.ipynb'
* '/MIOFlow/notebooks/[graphs]EB-traj-gene.ipynb'

2. C.elegans body wall muscle data
* '/MIOFlow/notebooks/[Graphs] Worm-Body-traj-gene.ipynb'

## Download generated cell trajectories:
1. [EB data]()

2. [C.elegans body wall muscle data](https://drive.google.com/file/d/1MBVpIC60f3bzHw_7uOYVgq2rSU3ecN-4/view?usp=share_link)

## How to run training script
(need to download generated cell trajectories, see train.py for more arguments detail)
```
python train.py --data eb|worm
```
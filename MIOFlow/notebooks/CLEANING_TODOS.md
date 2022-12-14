# TODO
- 1: skip load dataset section
- 2: find and implement all changes specified by TODO
- 3: clean notebook by removing sections / code marked for removal
- 4: use as template to recreate "good" notebooks in a more managable manner
- 5: update plot to have new legends (see slack channel)

# NEW TODOS
- 1. [DONE] move dyngen preprocessing into `07_datasets.ipynb`
- 2. [DONE] remove TODO comments / commented out old code
- 3. [DONE] commit, copy notebook, and then remove not-tweaked params
- 4. [Done] figure out which notebooks to keep / remove

# Notebooks
- [DONE] 1NN_metric @Sumner remove top part, keep just tabulation
- aml_pca, remove but add aml function to `07_datasets.ipynb` to load data, then copy clean notebook, and replace dyngen with aml
- dyngen_good is base for this notebook, confirm this notebook once done produces same results, then this can be replacement for Dyngen_good
- [Done] dyngen_tjnet_dami, @sumner, remove old code keep TJNet v1 as example of how to run since these were used in paper (add link to achive paper from Alex) rename TJNet  example
- [Done] EB-Bodies was PCA from Manik, bad labeling, fine to delete
- EB-fig-5-table-1 keep locally / pub branch @Guillaume eb_velocity_v5 file and backward_trajectories.npy and put in data/fig_1 dir for records (for reproducibility)
- [Done] EB_old (both) are fine to delete
- [Done] Jacks / Ring of Rings move to a new notebook just to show what Datasets exist
- @Guillaume petal_good, same as dyngen good, copy cleaned NB and reproduce results then delete old notebook
- Schiller @sumner, similar to dyngen, but for schiller data (and add schiller data to repo)
- SDE_test main difference is `which='sde'` in `make_model` function. We used in supplement. @Guillaume make sure that new notebooks runs with `which="sde"` and then this old notebook can be delete and replaced with a comment noting this feature's existance.

# 6/21/2022 TODOs
- update legend in plot
- update full param notebook with new train function
- rename and remove parital param notebook (keep the one with [sumner])
- aml, schiller, sde, petal_good can all be ported to new notebook schema
- get genes for EB fig 5 reproducibility
- future todo clean up datasets (where did it come from and how was it processed)

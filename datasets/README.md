# Datasets

## STATS SportVU NBA Dataset

The dataset comes from the player tracking data provided by 
[STATS SportVU](https://www.statsperform.com/team-performance/basketball/). 
The data are recorded with a series of cameras that surround the court at 360 degrees and give back a bird-eye view of 
players’positions. Each play composes of 50 time-steps sampled at 5Hz, where each time-step contains the positions 
(expressed as (x,y,z) world coordinates) for all the 10 players on the court (5 attackers, 5 defenders) plus the ball.

The dataset can be downloaded [here](https://www.dropbox.com/s/reibyhs7wmeoqc1/bsk_all_data.zip?dl=1) and comes in the
 ``bsk_all_data.zip`` archive. The zip contains the original dataset as a single numpy array ``all_data.npy``: once 
 extracted the archive content, put the .npy inside ``datasets/basket``. Then run
 ```
 cd datasets/scripts
 python preprocess_bsk.py --players [ atk | def | all ]
 ```     
to preprocess the data. This will create the ``datasets/bakset/<players>/<split>`` directories with the correct .npy 
splits that will be read by the PyTorch custom loader. 
 
  
Since preprocessing the data when loading the dataset introduces a time overhead, 
we decided to already save in advance all the needed (preprocessed) information inside the .npy splits, goals included: 
the custom PyTorch loader will only have to read the arrays and do nothing more. Saving all the information in advance 
also means that if you wish to change anything from the granularity of the goals 
grid to the percentages of the splits, you have to preprocess again the dataset and produce the corresponding .npy 
splits. If this is the case, run again the above script changing the arguments and/or the global variables that you need.  

In the given splits, by default, goals are computed fixing a 9x10 grid of 5x5 ft² cells (the half court is assumed 45x50 ft²).  

## Staford Drone Dataset
The *Stanford Drone Dataset* is composed of a series of top-down videos recorded by a hovering drone in 8 different 
college campus scenes. The datasets collects complex and crowded scenarios with various types of interacting targets:
we focus only on pedestrians, still their trajectories are influenced by the moving vehicles and subjects around them. 
We used the [TrajNet benchmark](http://trajnet.stanford.edu/) version of the dataset: trajectories are composed of 
a series of consecutive positions expressed as (x,y) world coordinates and recorded at 2.5FPS.  

The dataset can be downloaded [here](https://www.dropbox.com/s/owcynvz11tsq829/sdd_all_data.zip?dl=1) and comes in a
 ``sdd_all_data.zip`` archive. The zip contains a directory with single .txt files from the single scenes. 
 Put the ``all_data`` directory inside ``datasets\sdd``; then run 
  ```
  cd datasets/scripts
  python preprocess_sdd.py
  ```    
to preprocess the data. This will
 1. read all the trajectories from the single .txt files
 2. group them in a single .npy array
 3. produce the .npy splits from the single array
 4. save the splits in the ``datasets/sdd/sdd_npy`` directory
 
On the contrary of what said for the NBA dataset, the other SDD data preparation phases (like the ground-truth goals 
extraction) are quite light, therefore are performed inside the PyTorch custom loader when loading the data.

I.e. if you wish to change the granularity of the goals grid or the size of the goals window, you do not need to 
preprocess again the data but you can directly change the arguments of the corresponding training scripts (for more info
on the scripts args, see [models/README.md](../models/README.md)).

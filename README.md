# TS-LPP
Two Step Locality Preserving Projections (TS-LPP)

# Requirements
Python >= 3.7

numpy >= 1.16.2

scikit-learn >= 0.20.3

scipy >= 1.4.1

matplotlib >= 2.2.3

# Installation
Download or clone the github repository, e.g. git clone https://github.com/rtmr/TS-LPP

# Usage

## Parameters

**n_components: int** (The reduced dimension in the final step)

**knn: int** (The number for neighbor graph in LPP)

**n_clusters: int** (Number of target clusters)

## Target dataset

Target dataset is set to the feature matrix without label information.
(Target.csv contains the LAAF descriptors for solid, liquid, and amorphous states in silicon.)

## Execution
```
python TS-LPP.py 
```

# License
This project is licensed under the terms of the MIT license.

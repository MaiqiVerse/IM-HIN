# Introduction

The code respository for MICCAI submission "Integrating 3D Structures and Heterogeneous Information Network for Protein-Phenotype Prediction".

# Requirements

- Python 
- Pytorch (with CUDA support)
- json
- numpy
- pandas
- sklearn
- scipy

# Data Processing

You could follow the guidelines in [the code repository](https://github.com/liulizhi1996/HPODNets) to process the data, including running the `split_dataset_cv.py` script to produce hp.pkl, and processing each PPI network. 

Put all produced files into one folder, including hp.pkl, STRING_v12_filtered.json, genemania_filtered.json, humannet_xn_filtered.json.

## 3D Structural Data

Download pdb files for all proteins from [AlphaFold Database](https://alphafold.ebi.ac.uk/). You can convert the x, y, z index into txt (follow `protein_features/convert_pdb.py`). Then, use [code respository for pointnet++](https://github.com/charlesq34/pointnet2) to extract the 3D structural features. You need to put `protein_features/pointnet_processed.py` into pointnet++ folder, and run it. Notice that you need to change the path to where you put txt files.

Modify the path in the function `structure_feature_process` in load_dataset.py to where you have produced the embeddings from pointnet++.

# Run

After ajusting the paths in main.py, run:

```
python main.py
```



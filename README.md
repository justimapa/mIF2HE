# mIF2HE

This repository contains jupyter notebook files to create a database of labeld H&E patches that can then be used to train deep learning models.

## Cell_Mapping_MELANOMA_3.ipynb
Jupyter notebook that allows the registration of a mIF image with the equivalent H&E image. It uses mIF cell positions that will then be transformed to match the H&E file. These positions are compared to the output of the H&E segmenation using the HoverNetmodel which filters out low-quality annotations from the mIF image. At the end, one can save the positions so they can be used later to extract patches and store them in a database.

## Cell_Mapping_several_tiles.ipynb
A condensed version of the Cell_Mapping_MELANOMA_3.ipynb notebook allowing the registration of a full WSI slide.

## Data Processing.ipynb
Jupyter notebook that allows the user to extract labeled patches of cell, and save them directly in a database.

## DenseNet.ipynb
Jupyter notebook that allows the user to train a DenseNet model on labeled patches of cell.

## Contributors

[Antoine Ribault Gaillard](antoine.ribaultgaillardc@gmail.com) , [Justin Chrisitian Mapanao](justin.mapanao@hotmail.com)

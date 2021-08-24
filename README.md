# mIF2HE

This repository contains 3 major jupyter notebook files to create a database of labeld H&E patches that can then be used to train deep learning models.

###Cell_Mapping_MELANOMA_3.ipynb
Jupyter notebook that allows the registration of a mIF image with the equivalent H&E image. It uses mIF cell positions that will then be transformed to match the H&E file. At the end, one can save the positions so they can be used later to extract patches and store them in a database

###Data Processing.ipynb
Jupyter notebook that allows the user to extract labeled patches of cell, and save them directly in a database.

###DenseNet.ipynb
Jupyter notebook that allows the user to train a DenseNet model on labeled patches of cell.

Do not hesitate to reach the author for questions:

antoine.ribaultgaillardc@gmail.com

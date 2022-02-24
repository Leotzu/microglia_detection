# Microglia Detection

**Authors:** Leo Mckee-Reid, Scott Howard, Elliot Lupini

**Description:** A deep convolutional neural network that trains on flouresence microscopy image data in order to automatically detect microglia cells. In collaberation with the Tremblay Lab at the University of Victoria.

**Folder structure:**

```.
├── 00_practice
│   └── simple_pipeline.py
├── 01_data
│   ├── 008
│   │   ├── trainingset_01
│   │   │   └── README.txt
│   │   ├── SLM.roi
│   │   └── SR.roi
│   ├── tiles
│   │   ├── 008-500-3000.png
│   │   ├── 008-5000-500.png
│   │   ├── 010-2000-0.png
│   │   ├── 010-2500-2500.png
│   │   ├── 022-1500-9500.png
│   │   ├── 022-3000-7500.png
│   │   ├── 089-1500-500.png
│   │   ├── 089-3000-1000.png
│   │   ├── 121-1000-2500.png
│   │   ├── 121-2000-2500.png
│   │   ├── 121-2000-6000.png
│   │   ├── 121-2500-3000.png
│   │   ├── 121-2500-8000.png
│   │   ├── 121-3500-9500.png
│   │   └── 121-500-6000.png
│   ├── basic_gp_training.py
│   ├── convert_data.py
│   ├── create_zarr.py
│   └── simple_pipeline.py
├── 02_train
│   ├── gunpowder_tutorials
│   │   └── simple_pipeline.py
│   ├── setup01
│   │   ├── results
│   │   │   └── 008
│   │   │       └── trainingset_01
│   │   ├── train.py
│   │   ├── train_old.py
│   │   └── train_temp.py
│   └── setup02
│       └── train.py
├── 03_view
└── create_tree.py```

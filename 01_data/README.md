# 01_data

This is where we store all the data as well as do all the preprocessing.

## Downloading data from dropbox

For each dataset you want to begin working with, perform the following commands within this directory:
* mkdir \<dataset name\>
* cd \<dataset name\>
* wget -O data.zip \<dropbox link\>
* unzip data.zip

#### Dataset names and associated dropbox links:

* dataset name: 008
  * dropbox link: https://www.dropbox.com/sh/hx73hvnbdlukzt9/AADkmgK9f2JwYgxnrOTHwaX0a?dl=0
* dataset name: 010
  * dropboax link: https://www.dropbox.com/sh/dux9zp87ie3rxkr/AAB1mdwHvfGw-7AdceXOwl_Ma?dl=0
* dataset name: 022
  * dropbox link: https://www.dropbox.com/sh/3t1aqdhxhk5nyfe/AADbQ2qPa5R67P2M5vd2YD9ha?dl=0
* dataset name: 089
  * dropbox link: https://www.dropbox.com/sh/ltepog62oefnnz6/AAAHz9ibMHjWQNFBYZBXAmdLa?dl=0
* dataset name: 121
  * dropbox link: https://www.dropbox.com/sh/c4tipo0m482x5j0/AAAo-H8XsjRYF8Ipa6VjEI7Na?dl=0

## Image tiles

A random sample of the dataset was produced to be manually classified. The methodology for selecting the random sample is described as follows:

1. Each of the 5 input images (listed above) was "tile-ized" into 500x500 images without overlap. The output images were named as follows: AAA-B-C.png, where AAA is the 3-digit dataset name, as listed above; B is the X offset from the left side of the source image; and C is the Y offset from the top of the image.
2. A histogram of each image was produced to identify the number of white pixels (the area of the source image outside of the microscopy data defaults to white), and filter out the ones that had no, or little, data. >80% white pixels was determined to be "no data", and 25-80% was taken to be "too little data." There were 420 suitable image tiles source from the 5 input datasets.
3. A random sample of 15 images was created using python's [`random.sample()`](https://docs.python.org/3/library/random.html#random.sample) routine.

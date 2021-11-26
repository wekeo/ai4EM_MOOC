# JupyterLab-WEkEO

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/wekeo/ai4EM_MOOC/graphs/commit-activity)
![python](https://img.shields.io/badge/python-3.9-blue)


Throughout the [Artificial Intelligence for Earth Monitoring MOOC](https://www.futurelearn.com/courses/artificial-intelligence-for-earth-monitoring), there is hands-on tutorials where you can have a go yourself at using AI and ML algorithms to analyse Earth observation data (from the Copernicus programme). These practical elements of the course will take the form of Jupyter Notebooks, which work like documents containing both rich-text (such as paragraphs and links) and computer code, in this case Python, and the code can be run which will output results.

### What topics will you cover?

* An introduction to artificial intelligence (AI) and machine learning (ML) in the EU Copernicus Programme.
* Accessing Earth observation data and working with it through the WEkEO platform.
* An overview of AI and ML types, technology, and terminology.
* How AI and ML are used in various EO applications.
* Hands-on learning using the Python programming language and Jupyter Notebooks to process and analyse EO data using ML.

## Getting Started

### Description

**ai4EM_MOOC** is a repository of Python based tools to introduce you to the [WEkEO DIAS (Data Information and Access System)](https://wekeo.eu/) and the new cutting-edge AI algorithms. Within the submodules of this repository are tutorials and case studies, using data from the Copernicus Programme that are available on WEkEO and written by expert trainers. This course is funded by the Copernicus Programme and has been put together by EUMETSAT, ECMWF, Mercator Ocean International and the EEA.

The experts in AI, EO, and Earth system monitoring will take you through the importance of AI in [Week 2](https://github.com/wekeo/ai4EM_MOOC/blob/main/2_ai4eo) and in four themed weeks – land ([Week 3](https://github.com/wekeo/ai4EM_MOOC/blob/main/3_land)), ocean ([Week 4](https://github.com/wekeo/ai4EM_MOOC/blob/main/4_ocean)), atmosphere ([Week 5](https://github.com/wekeo/ai4EM_MOOC/blob/main/5_atmosphere)), and climate ([Week 6](https://github.com/wekeo/ai4EM_MOOC/blob/main/6_climate)) – leaving you well-versed in the intricacies of EO and satellite data, as well as how AI and ML can unlock its full potential. 

**NOTE**: It is important to know that the following GitHub repository **ONLY** contains the Jupyter notebooks **WITHOUT** the data utilized for the notebooks. The data that is needed for each week can be found below:

* [Week 2](https://wekeo-files.apps.mercator.dpi.wekeo.eu/s/iqMwt7Zz8isxg2K)
* [Week 3](https://wekeo-files.apps.mercator.dpi.wekeo.eu/s/A2wnqLpAAHnKByt)
* [Week 4](https://wekeo-files.apps.mercator.dpi.wekeo.eu/s/teTmKxmA2eSMssp)
* [Week 5](https://wekeo-files.apps.mercator.dpi.wekeo.eu/s/E2prXaiWQNXWrbS)
* [Week 6](https://wekeo-files.apps.mercator.dpi.wekeo.eu/s/FjK8Z9tyzjnWRKX)

In order to have the same structure as the `public` folder, each folder data needs to be moved to its specific week folder. 

### Installing the Python environment

This repository supports Python 3.9. We highly recommend that users working on their own systems install the appropriate Anaconda distribution for their operating system. 

The Python environment ("machine-learning") that need to be utilized for executing the Jupyter notebooks can be found [here](https://github.com/wekeo/ai4EM_MOOC/blob/main/env.yaml). The [env.yaml](https://github.com/wekeo/ai4EM_MOOC/blob/main/env.yaml) file is utilized to load all the Python packages needed.

Additionally, a [Docker](https://github.com/wekeo/ai4EM_MOOC/blob/main/Dockerfile) file ([Dockerfile](https://github.com/wekeo/ai4EM_MOOC/blob/main/Dockerfile)) can be used to run the Python environment. 

### Cloning the repository

#### Own local machine

To clone this repository in your laptop, type:

``` shell
    git clone https://github.com/wekeo/ai4EM_MOOC.git
```

Then, for all the Jupyter notebooks that utilize the Snippet, the importing code cell needs to changed **MANUALLY**. The code lines that can be found below should be removed when it is executed in your own local machine:

```
## BEGIN S3FS IMPORT SNIPPET ##
import os, sys
s3_home =  os.getcwd()
try: sys.path.remove(s3_home) # REMOVE THE S3 ROOT FROM THE $PATH
except Exception: pass

current_dir = os.getcwd()

os.chdir('/home/jovyan') # TEMPORARILY MOVE TO ANOTHER DIRECTORY

# BEGIN IMPORTS #

# END IMPORTS #

os.chdir(current_dir) # GO BACK TO YOUR PREVIOUS DIRECTORY

sys.path.append(s3_home) # RESTORE THE S3 ROOT IN THE $PATH

## END S3FS IMPORT SNIPPET ##
```

**NOTE**: You would not have the fast access provided by the Harmonized Data Access as part of the WEkEO infrastructure if you want to execute in your own local machine.

#### JupyterHub

In case you want to copy the repository in a local folder of your Gitlab, you would need to register for a [WEkEO](www.wekeo.eu) account and enter the JupyterHub - then follow the instructions below.

If you are currently on the WEkEO JupyterLab you are already in the right place and can start. To clone this repository in to the WEkEO JupyterLab environment open a terminal in the WEkEO JupyterLab, type:

``` shell
    cd work
    git clone https://github.com/wekeo/ai4EM_MOOC.git
```

This will create a clone of this repository of notebooks in the work directory on your JupyterHub instance. You can use the same shell script to clone any external repository you like.


## Help

In case a problem or issue related to code is found, please create a new issue in GitHub, or post a comment on FutureLearn's forum.

If the problem is related to WEkEO, please contact the WEkEO support by requesting a ticket ([support@wekeo.eu](support@wekeo.eu)).

## License

This code is licensed under an MIT license. See file LICENSE.txt for details on the usage and distribution terms.

All product names, logos, and brands are property of their respective owners. All company, product and service names used in this website are for identification purposes only.
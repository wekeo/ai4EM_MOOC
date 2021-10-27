# JupyterLab-WEkEO

Throughout the [Artificial Intelligence for Earth Monitoring MOOC](https://www.futurelearn.com/courses/artificial-intelligence-for-earth-monitoring), there is hands-on tutorials where you can have a go yourself at using AI and ML algorithms to analyse Earth observation data (from the Copernicus programme). These practical elements of the course will take the form of Jupyter Notebooks, which work like documents containing both rich-text (such as paragraphs and links) and computer code, in this case Python, and the code can be run which will output results (such as graphs and tables).

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

### Installing the Python environment

This repository supports Python 3.9. We highly recommend that users working on their own systems install the appropriate Anaconda distribution for their operating system. 

The Python environment ("machine-learning") that need to be utilized for executing the Jupyter notebooks can be found [here](https://github.com/wekeo/ai4EM_MOOC/blob/main/env.yaml). The [env.yaml](https://github.com/wekeo/ai4EM_MOOC/blob/main/env.yaml) file is utilized to load all the Python packages needed.

Additionally, a [Docker](https://github.com/wekeo/ai4EM_MOOC/blob/main/Dockerfile) file ([Dockerfile](https://github.com/wekeo/ai4EM_MOOC/blob/main/Dockerfile)) can be used to run the Python environment. 

### Cloning the repository

To clone this repository in your laptop, type:

``` shell
    git clone https://github.com/wekeo/ai4EM_MOOC.git
```

NOTE: You would not have the fast access provided by the Harmonized Data Access as part of the WEkEO infrastructure if you want to execute in youe own local machine.

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

The license is defined in each Jupyter notebook.
### Description:
This project explored the application of conventional supervised 
approaches to human activity recognition and the application of an 
Isomap based domain adaptation method for transfer learning between 
individuals. This work also presents a modification of the approach and 
shows the modification to perform better than the original domain 
adaptation method on this dataset. Code in this repository implements
the transfer learning approaches attempted. Work was part of the 
COMPSCI-589 course project.
    
### Dataset:
Link: http://sensor.informatik.uni-mannheim.de/#dataset_realworld

### Work implements:
Chattopadhyay, Rita, Narayanan Chatapuram Krishnan, and Sethuraman 
Panchanathan. "Topology Preserving Domain Adaptation for Addressing Subject 
Based Variability in SEMG Signal." AAAI Spring Symposium: Computational 
Physiology. 2011.
Link: https://pdfs.semanticscholar.org/881d/78226a8e05bb4cad13d4e833f5a73a61eb77.pdf

### Summary of findings of project:
Link: https://www.dropbox.com/s/lnhgwzogguiz5gz/COMPSCI589-Report.pdf?dl=0

### Description of directories and files:
`data`: Extracted features stored as pandas dataframes serialized to 
    pickle files.

`data_utils/settings.py`: Global settings used across whole project.

`data_utils/tsio.py`: Read data from complex directory structure of
    dataset and return it in an easy to work with format.

`data_utils/feature_extract.py`: Windowing on time series and extraction of features.

`data_utils/transform.py`: Utilities to read and structure data into 
    suitable dataframes.
    
`models/dom_adapt.py`: Impliments the transfer learning approaches.

`driver.py`: Runs all experiments and prints results to STDOUT.

### Note:
Work was tested on Ubuntu 14.04 and used routines from `scikit-learn`, 
`pandas` and `numpy`. Also, a lot more work could go into this; certainly 
engineering wise, maybe also conceptually. Will get to it, time permitting.

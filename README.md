# Cloud Project
Kevin Sherman, Alexandra DeLucia, Hadley VanRenterghem

## Getting Started

1. Clone this repo 

    ```bash
    git clone git@github.com:AADeLucia/cloud-project-fall2020.git
    ```

2. Download the dataset from [here](https://1drv.ms/u/s!AnuG4njwMZEOgWIomvPD4Nm_rpj-?e=Nh1AnE). This downloads an
 `ml` folder. Replace the `/flux/data/ml` folder with this `ml` folder.
 
3. Setup up the Python environment. We used python 3.6. The easiest way to set up this environment is with anaconda.
    ```bash
    conda create --name <name> python=3.6
    conda install pip
    pip install -r requirements.txt
    ```

    In order to use the `xgboost` module, you will need another library installed. The easiest way to install this library 
    is with Homebrew,
    
    ```bash
    brew install libomp
    ```

4. Update the paths in the scripts in `/scripts` to match your directory setup. Absolute paths are more reliable.


## Running the experiments

The scripts are in `/scripts`. Remember to activate your new python environment before running the scripts.

---

Please contact us if you have any issues. 

**Alexandra DeLucia:** aadelucia@jhu.edu

**Kevin Sherman:** ksherma6@jhu.edu

**Hadley VanRenterghem:** hvanren1@jhu.edu

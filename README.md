# Cloud Project
Kevin Sherman, Alexandra DeLucia, Hadley VanRenterghem

## Getting Started (with Google Cloud VM)

1. Email team (emails below) for access to Google Cloud VM.

2. Log into Google Cloud Platform and under `VM Instances` click `SSH` to access the VM.

3. Change to the tester user and move into project directory. You will be prompted for a password. The password is `tester`.

    ```bash
   su tester && cd ~/cloud-project-fall2020 
   ```

4. You are ready for results reproduction! Skip to "Running the experiments" section below.

## Getting Started (without Google Cloud VM)

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

All scripts to reproduce our results are in `/scripts`. 

To reproduce our reproduction of the original Flux models: 
```bash
./flux_ffn.sh
./flux_xgboost.sh
```

To run our baseline for using the average of the previous flows:
```bash
./linear_baseline.sh
```

To run our model robustness experiments:
```bash
./xgboost_robust.sh
```

The numbers will match the results on the slides.

---

Please contact us if you have any issues. 

**Alexandra DeLucia:** aadelucia@jhu.edu

**Kevin Sherman:** ksherma6@jhu.edu

**Hadley VanRenterghem:** hvanren1@jhu.edu

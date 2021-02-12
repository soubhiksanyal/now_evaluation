# now_evaluation
This is the official repository for evaluating NoW Benchmark Dataset

## Clone the repository 
```
git clone https://github.com/soubhiksanyal/now_evaluation.git
```
## Installation

Please install the virtual environment

```
mkdir <your_home_dir>/.virtualenvs
python3 -m venv <your_home_dir>/.virtualenvs/now_evaluation
source <your_home_dir>/.virtualenvs/now_evaluation/bin/activate
```

Install the requirements by using:

```
pip install -r requirements.txt
```

Install mesh processing libraries from MPI-IS/mesh within the virtual environment.

### Installing Scan2Mesh distance:

Clone the flame-fitting repository and copy the required folders by the following comments

```
git clone https://github.com/Rubikplayer/flame-fitting.git
cp flame-fitting/smpl_webuser now_evaluation/smpl_webuser -r
cp flame-fitting/sbody now_evaluation/sbody -r
```

Clone Eigen and copy the it to the following folder 

```
git clone https://gitlab.com/libeigen/eigen.git
cp <your_home_dir>/eigen <your_home_dir>/now_evaluation/sbody/alignment/mesh_distance/eigen -r
```

Compile the code in the directory 'now_evaluation/sbody/alignment/mesh_distance' by the follwing command

```
cd <your_home_dir>/now_evaluation/sbody/alignment/mesh_distance
make
```

The installation of Scan2Mesh is followed by the codebase provided by flame-fitting.
Please check that repository for more detailed instructions on Scan2Mesh installation.

## Evaluation

Go to the main directory 

```
cd <your_home_dir>/now_evaluation
```

Run 

```
python main.py
```

The function in metric_computation() in main.py is used to compute the error metric. Please change the paths in metric_computation() as following,

```
predicted_mesh_folder =  path to predicted mesh folder
imgs_list = path to the test or validation image list downloaded from the ringnet website

gt_mesh_folder =  path to the ground truth scans
gt_lmk_folder = path to the scan landmarks
```

Cumulative error curves from the computed error metric can be also generated from the main.py file.
The corresponding function is generating_cumulative_error_plots().

Please change the paths accordingly in generating_cumulative_error_plots().






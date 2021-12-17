# Recsys

Datasets: `MovieLens`, `Netflix`.
Build a Recommender system using Graph Neural Networks.


## How to use

`To analyze results`:
1. Get data folder
2. clone the repo
3. Check the artifacts folder for all data for each experiment

`To run and analyze`
1. Get data folder from `https://drive.google.com/file/d/11x3FsHz6JcqPNrF_ECHJO-zZnFzcX6vC/view?usp=sharing`
2. clone the repo
3. Install python 3.7 and above
4. Navigate to project folder and run below commands

create virtual env and install required packages -- run the below commands but remove 

you really don't need `wand`,so remove that line from file `requirements.txt`

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt


`To See the implementation and training details`

See (basic_implementation.ipynb)[https://github.com/CS-512-Recsys/Recsys/blob/main/nbs/basic_implementation.ipynb] notebook. It contains the code and training for the Deep Learning based recommender systems. For the baseline check ()



### Note: To get the movie-id,user-id for the predictions for test and valid check `partial_random_tst.csv` and `partial_random_vld.csv`. Using this info we can extract the corresponding genre from `tag.csv`.







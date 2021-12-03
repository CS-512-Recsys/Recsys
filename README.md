# Recsys

Datasets: `MovieLens`, `Netflix`.
Build a Recommender system using Graph Neural Networks.


## How to use

`To analyze results`:
1. Get data folder
2. clone the repo
3. Check the artifacts folder for all data for each experiment

`To run and analyze`
1. Get data folder
2. clone the repo
3. Install python 3.7 and above
4. Navigate to project folder and run below commands

### create virtual env and install required packages -- run the below commands but remove 

### you really don't need `wand`,so remove that line from file `requirements.txt`

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

# Note: To get the movie-id,user-id for the predictions for test and valid check `partial_random_tst.csv` and `partial_random_vld.csv`. Using this info we can extract the corresponding genre from `tag.csv`.




## WorkFlow (Notes to myself)


Phase 1: Barebones working
- Setup a clean environment - `recsys`,with python=3.9(as typehinting is inbuilt)
Checkout to `vin-dev`
- Setup `gitignore`(ignore `data` folder,etc)
- Install Packages and make sure to store(withversion) in requirements.txt
- Write Code in `jupyter notebook` for `getting data`,`cleaning data`,`transformations`, `Pytorch trainer loop` with `Dataset` and `Dataloader` abstractions.
- Make Samples of the data-sets and run complete-training and inference.
- Tools : Experiment tracking->`Weights and Biases`.

Phase 2: Make it Runnable(scripts and reproducablity)

- Convert notebook to script and refactor code into multiple files - (
- Edit in VSCODE With the extension to get the `docs` and also add `type hinting`
- Build a CLI , tool: [`TYPER`](https://typer.tiangolo.com)
- Make Sure the `artifacts` are tracked and pushed but not data.

Run it on complete data, get results and merge into `main` branch.
Add `GAT` Model and perform the same experiments.




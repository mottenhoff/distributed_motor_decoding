# Decoding executed and imagined grasping movements from distributed non-motor brain areas using a Riemannian decoder

Code used in:
> Ottenhoff MC, Verwoert M, Goulis S, Colon AJ, Wagner L, Tousseyn S, van Dijk JP, Kubben PL and Herff C (2023) - Decoding executed and imagined grasping movements from distributed non-motor brain areas using a Riemannian decoder. Front. Neurosci. 17:1283491. doi: 10.3389/fnins.2023.1283491

[Read the paper](https://www.frontiersin.org/articles/10.3389/fnins.2023.1283491/abstract)\
[Access the data](https://osf.io/xw386/)

## Installation

Tested for python 3.9, but other versions will likely work as well.

`git clone https://github.com/mottenhoff/distributed_motor_decoding.git`\
Download [the data](https://osf.io/xw386/) to the cloned repository folder. I used `./data` to save the data, but you can change the the folder in `decode.py:run() data_path`.


`conda create --name your_env_name python=3.9`\
`conda activate your_env_name`\
`pip install -r requirements.txt`\

## Running code

#### Decoder:

run `python main.py`

To run all files in parallel, set `PARALLEL` in `main.py` to True.\
`main.py:main()` also contains all the variables used (`exps`, `bands` and `ppts`).\
To change the included components, change the `n_components` in `decode.py:run()`.\
To run the CSP analysis, set DECODE_CSP_LDA in `decode.py:42` to `True`.\

#### Plot results

run `plot_figures.py`\
Set the path to the results in `plot_figures.py():run() -> path_results`

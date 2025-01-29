# README.md

# Graduation Thesis Code Snippets

This project implements an economic model simulation using Python. The model is designed to analyze and simulate economic behaviors based on various parameters and data inputs.

## Project Structure

```
economic-model
├── src
│   ├── main.py               # Entry point for the economic model simulation
│   ├── mbr_vars.py           # Defines global variables and parameters
│   ├── nr.py                 # Implements the Nelder-Mead method for minimization
│   ├── nrutil.py             # Contains utility functions for numerical routines
│   ├── fill_miles50.py       # Reads data and creates a distance matrix
│   ├── Like_i.py             # Calculates likelihood of individual histories
│   ├── evf.py                # Computes expected value function and choice probabilities
│   ├── func_mpi.py           # Contains subroutines for parallel computation using MPI
│   ├── par_values.py         # Prints parameter values and their standard errors
│   ├── f1dim_mod.py          # Module for one-dimensional search algorithms
│   ├── read_climate.py       # Reads climate data
│   ├── slave_work.py         # Handles work extraction tasks
│   ├── util_age.py           # Computes utility flows
│   ├── ass66.py              # Evaluates tail area of the standardized normal curve
│   ├── lubksb.py             # Solves linear equations using LU decomposition
│   ├── ludcmp.py             # Performs LU decomposition of matrices
│   ├── GIH.py                # Computes numerical approximations of gradients and Hessians
│   ├── Newton.py             # Implements parameter optimization using line search
│   ├── read_adj.py           # Reads adjacency data and fills adjacency matrix
│   ├── read_data.py          # Reads and processes data files
│   ├── mbr93a_MPI.py         # Header for the Fortran program with historical info
│   ├── start_values.py       # Initializes parameter indices
├── data
│   └── cfps10_22mc.dta       # Data used in the model
├── requirements.txt           # Lists project dependencies
└── README.md                  # Project documentation
```

## Installation

To run this project, ensure you have Python installed along with the required dependencies. You can install the dependencies using:

```
pip install -r requirements.txt
```

## Usage

To execute the simulation, run the following command:

```
python src/main.py
```

This will initialize the parameters, read the necessary data, and execute the main functions of the economic model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
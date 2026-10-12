# Shrinking Cities Ecological Trajectories

Analysis code for studying ecological trajectories and their driving mechanisms in shrinking cities in China.

## Contents

- **Piecewise SEM:** R code for baseline model estimation and temporal–spatial robustness analyses, including AR(1) correction, residual Moran’s I tests, and Moran’s eigenvector map (MEM) filtering.
- **Bayesian spatiotemporal changepoints:** Python code for a hierarchical model that jointly estimates changepoint occurrence and timing, including an explicit no-change state.

## Requirements and Usage

The PSEM workflow requires R and the packages listed in `install_packages.R`. The Bayesian model requires Python 3.10 or later and the dependencies listed in `pyproject.toml`.

Installation instructions, input-data requirements, and execution commands are provided in the README within each folder.

## Data Availability

Research data and analysis outputs are excluded from this repository. Data access is described in the associated article’s Data Availability Statement.

## License

The code is available under the MIT License.

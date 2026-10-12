# Piecewise SEM analysis code

This repository contains only the R code and the fixed MEM-filter specification
used for the Piecewise SEM analyses. 

## Local input files

To run the analysis, create a local `data/` directory and add:

- `data/psem_panel.xlsx`
- `data/city_coordinates.xlsx`


## Run

```bash
Rscript install_packages.R
Rscript run_all.R
```

Individual analysis scripts are stored in `R/`. The file
`config/mem_filter_published.csv` records the fixed year-specific MEM terms
needed to reproduce the reported spatial-filtering specification.

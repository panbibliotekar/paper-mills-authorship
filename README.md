# Paper mills and the changing structure of scientific authorship

This repository contains the Python code used for the analysis reported in the manuscript:

**Paper mills and the changing structure of scientific authorship**

The study examines authorship patterns in paper mill-related retractions indexed in the Retraction Watch Database between 2000 and 2024.

## Data source

The analysis uses the Retraction Watch Database, available from Crossref:

https://gitlab.com/crossref/retraction-watch-data

The dataset is not redistributed in this repository. To reproduce the analysis, download the CSV file from the official source and place it in the project folder as:

```text
retraction_watch.csv

## Reproducing the analysis

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the analysis script:

```bash
python scripts/paper_mill_authorship_analysis.py
```

The script generates summary tables and figures in the `outputs/` directory.

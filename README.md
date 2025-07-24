# Distantly Supervised Pyramid Networks

![DSPN](logo.png)

This repository contains code for training and evaluating our proposed distantly supervised pyramid network for Unified Sentiment Analysis.

## Overview

We introduce Unified Sentiment Analysis, a novel framework integrating aspect detection, sentiment analysis, and rating prediction. 

We also propose the Distantly Supervised Pyramid Network (DSPN), which hierarchically models sentiment using only rating labels, as an implementation of unified sentiment analysis. Experiments show DSPN matches benchmark performance while improving efficiency and interpretability.

## Repository Structure

```
📂 data/processed             # Contains processed datasets
📂 preprocess/            # Notebook for preprocessing datasets
📜 DSPN.ipynb  # Main notebook for DSPN experiments
📜 Packages.ipynb  # Notebook for installing required libraries
📜 logo.png     # LOGO of DSPN
📜 models.py         # Code for model implementation
📜 preprocess.py         # Preprocessing code for data input
📜 test_func.py         # Functions for testing model performance
📜 trainer.py         # Code for model training
📜 utils.py         # Common code shared across files
📜 LICENSE               # Project license information
📜 README.md               # Project documentation
```

## Setup & Installation

Please refer to the Packages.ipynb notebook.

## Usage

You can train and test DSPN by running the DSPN.ipynb notebook. The model can also be adjusted for customization in models.py.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
# Contributors

This project is maintained by the Human-centered Analytics Lab (HAL).

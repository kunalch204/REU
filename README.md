# Comparing the Performance of Different Forms of Human Feedback for Safe Reinforcement Learning

## Introduction

This project is part of a Research Experience for Undergraduates (REU) at Oregon State University. It examines how different types of human feedback can enhance the safety and accuracy of reinforcement learning (RL) models. By addressing data imbalances and incorporating human insights, we aim to improve the predictability and reliability of RL systems.

### Project Details

- **Author**: Kunal Chopra
- **Principal Investigator**: Sandhya Saisubramanian
- **Mentor**: Yashwanthi Anand
- **Institution**: Oregon State University
- **Date**: 9/14/2023

Our research investigates the impact of human feedback on the learning outcomes of RL models, focusing on creating safer and more effective AI systems through experimental analysis and comparative studies.

The project is rooted in the understanding that the balance and quality of training data are critical to the effectiveness of predictive models in machine learning. By integrating human feedback into the RL process, we aim to counterbalance the skewness of imbalanced datasets and enhance the overall quality of model predictions, paving the way for future strategies that could leverage human insight for optimal RL outcomes.

## Getting Started

To replicate our findings or to build upon our research, follow the steps outlined in this section to set up the project environment on your local machine.

### Prerequisites

Ensure you have the following prerequisites installed:
- Python 3.x
- Git (for cloning the repository)

### Installation

Setting up this project locally involves a few simple steps. Begin by cloning the repository, and then proceed based on your preference for virtual environments or Conda environments.

1. **Clone the Repository**

    Get started by cloning this repository to your local machine. Open a terminal and run the following commands:

    ```shell
    git clone https://github.com/yashianand/HF-performance-analysis.git
    cd HF-performance-analysis
    ```

    This will create a local copy of the repository and navigate you into the project directory.

2. **Setting Up a Virtual Environment (Recommended for most users)**

    Using a virtual environment for your Python projects helps manage dependencies cleanly. If you have `virtualenv` installed, set up your environment with these commands:

    ```shell
    virtualenv myenv
    . ./myenv/bin/activate
    pip install -r requirements.txt
    ```
    
3. **Setting Up a Conda Environment (Recommended for Anaconda/Miniconda users)**

    If you prefer using Conda environments, especially when working with Anaconda or Miniconda, you can set up your environment with the following commands:

    ```shell
    conda create -n myenv python
    conda activate myenv
    pip install -r requirements.txt
    ```

    This creates a new Conda environment named `myenv`, activates it, and installs the necessary packages from the Conda-Forge channel.

After following these steps, your development environment will be ready, and you can proceed to run or modify the project as needed.

## Dataset Generation

This project is divided into two main phases, each with its unique approach to dataset generation and analysis. Here's an overview of what each phase entails:

### Phase 1: Grid World Datasets

In Phase 1, we generated datasets using grid world designs, a common framework in RL experiments for understanding basic RL concepts. The datasets are created through scripts adapted from the open-source MiniGrid library, with significant contributions from Yashwanthi Anand for this study.

- **Objective**: The goal was to analyze the entropy of various datasets to understand data balance and its impact on prediction outcomes.
- **Implementation**: We utilized scripts that simulate different grid setups (sizes 6 and 17) to study RL dynamics, focusing on state-action pairs and their influence on agent decision-making.

### Phase 2: Atari "Freeway" Game Feedback

Phase 2 extends the analysis to real-world human feedback data collected from the Atari game "Freeway." This phase aims to understand the entropy dynamics in datasets enriched with human interactions.

- **Objective**: To calculate the entropy values for human feedback data and explore patterns in user responses during gameplay.
- **Data Source**: The feedback data, including general responses and those with human gaze information, was obtained from [Zenodo](https://zenodo.org/record/3451402/files/freeway.zip?download=1) and further processed for analysis.

## Usage

This project facilitates dataset generation and analysis for both Phase 1 (Grid Size 6 and Grid Size 17) and Phase 2 (Atari "Freeway" game feedback) through dedicated scripts.

### Running Phase 1 and Phase 2

To execute dataset generation and analysis for Phase 1 with Grid Size 6 and Phase 2, use the `run.py` script:

```shell
python run.py
```

### Running Phase 1 with Grid Size 17 

```shell
python run17.py
```

This initiates the dataset generation and analysis for the larger grid size of 17 in Phase 1.

## Analyzing Results

After completing the dataset generation and analysis for both phases, you can review the detailed statistics and outcomes in the generated CSV files, located in the same directory as the `run.py` and `run17.py` scripts.

### Phase 1 Results

For Phase 1, the analysis results for both Grid Size 6 and Grid Size 17 are available in the following files:

- `Phase_1_final_results_6.csv`
- `Phase_1_final_results_17.csv`

These files contain detailed statistics including entropy, accuracy, and F1 score for each form of feedback analyzed.

### Phase 2 Results

For Phase 2, which involves analyzing human feedback from the Atari "Freeway" game, the results are provided in the following CSV files located in the same directory as the `run.py` script:

- `DEMO_dataset_predictions_results.csv`
- `DEMO_dataset_predictions_summary.csv`
- `DEMO_entropy_values.csv`

#### Individual Model Performance

The `DEMO_dataset_predictions_results.csv` file details the performance of models tested on individual datasets. Each row in this file represents a model, the dataset it was tested on, and the corresponding Accuracy and F1 Score metrics.

#### Summary Statistics

The `DEMO_dataset_predictions_summary.csv` file includes average and standard deviation values for accuracy and F1 scores across all models and datasets.
This file provides a high-level overview of the models' effectiveness in interpreting human feedback within the "Freeway" game context, offering insights into the average performance and variability across different feedback types.

#### Entropy Analysis

The `DEMO_entropy_values.csv` file contains the entropy values calculated for the feedback datasets, offering insights into the unpredictability and information content inherent in human feedback.
This file is crucial for understanding the diversity and complexity of the feedback data. Higher entropy values indicate greater unpredictability in the feedback, which can significantly influence the learning dynamics of RL models.

## Acknowledgments

I would like to extend my sincere thanks to everyone who has contributed to the success of the "Comparing the Performance of Different Forms of Human Feedback for Safe Reinforcement Learning" project:

- **Oregon State University**: For providing the resources and environment conducive to conducting this research.
- **Principal Investigator Sandhya Saisubramanian**: For her invaluable guidance and insights throughout the project.
- **Mentor Yashwanthi Anand**: For her mentorship and support in developing the methodologies and scripts used in this study.
- **Open Source Communities**: Gratitude to the developers and maintainers of the software and libraries I relied on, including but not limited to Python, NumPy, scikit-learn, and MiniGrid.

This project would not have been possible without the collective efforts and support of these individuals and communities.

from feedback.value_iteration import valueIteration
from feedback.helper_functions import *
from feedback.single_feedback import single_fb
from feedback.multiple_feedback import multiple_fb
from entropy import (
    get_csv_paths,
    calculate_entropy,
    multi_extract_entropy,
    manipulate_csv,
    extract_column,
    update_iteration,
    demo_get_entropy,
    extract_entropies_17,
    update_iteration_17,
)
from train import (
    clean_data,
    train,
    predict,
    convert_txt,
    demo_train,
    demo_predict,
    train_17,
    predict_17,
)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


np.set_printoptions(linewidth=500)


"""
-----------------INITIALIZE ENVIRONMENT VARIABLES-----------------------------

train_game: (needn't change this value)
    - Environment to train the agent
    - Always set it to 0. Oth grid in the gridArt file is the training grid

test_game: (set this value to choose different test grids)
    - Environment to test the agent in
    - Options: 1, 2, 3

gridSize: (set this value to choose different grid sizes)
    - Options: 6, 17
    - 6 is mostly used for debugging
"""
train_game = 0
test_game = 1
gridSize = 17

trainGrid = initialize_env(gridSize, train_game)


# ------------------INITIALIZE EXPERIMENT VARIABLES---------------------------

if gridSize == 6:
    iterations = [10, 20, 40, 80, 160, 200]
elif gridSize == 17:
    iterations = [400, 800, 1200, 1600, 2000, 2500]


experiment_name = (
    "entropy_test_25"  # Change this to a different name for each experiment
)
output_dir = "output/gridSize_" + str(gridSize) + "/" + experiment_name + "/"

testset_filename_one = output_dir + "testSets/game" + str(1) + ".txt"
testset_filename_two = output_dir + "testSets/game" + str(2) + ".txt"
testset_filename_three = output_dir + "testSets/game" + str(3) + ".txt"
testset_filename_four = output_dir + "testSets/game" + str(4) + ".txt"


# ---------------------------GENERATE TRAIN SET-------------------------------

# Get oracle policy for trainGrid
_, trainGrid_oracle_policy = valueIteration(trainGrid, is_oracle=True)

# Get single feedback - one of DAM, Approval or Correction
single_fb(trainGrid, trainGrid_oracle_policy, iterations, output_dir)

# Get multiple feedbacks - combination using any 2 of DAM, Approval and Correction or all 3
multiple_fb(trainGrid, trainGrid_oracle_policy, iterations, output_dir)

# ---------------------------GENERATE TEST SET--------------------------------


feedback_types = [
    "3 multiFB/c-d-a/App",
    "3 multiFB/c-d-a/DAM",
    "3 multiFB/c-d-a/Corr",
    "2 multiFB/a-c/App",
    "2 multiFB/a-c/Corr",
    "2 multiFB/c-a/App",
    "2 multiFB/c-a/Corr",
    "3 multiFB/a-d-c/App",
    "3 multiFB/a-d-c/DAM",
    "3 multiFB/a-d-c/Corr",
    "3 multiFB/d-c-a/App",
    "3 multiFB/d-c-a/DAM",
    "3 multiFB/d-c-a/Corr",
    "2 multiFB/d-c/DAM",
    "2 multiFB/d-c/Corr",
    "2 multiFB/a-d/App",
    "2 multiFB/a-d/DAM",
    "3 multiFB/c-a-d/App",
    "3 multiFB/c-a-d/DAM",
    "3 multiFB/c-a-d/Corr",
    "3 multiFB/d-a-c/App",
    "3 multiFB/d-a-c/DAM",
    "3 multiFB/d-a-c/Corr",
    "2 multiFB/c-d/DAM",
    "2 multiFB/c-d/Corr",
    "3 multiFB/a-c-d/App",
    "3 multiFB/a-c-d/DAM",
    "3 multiFB/a-c-d/Corr",
    "2 multiFB/d-a/App",
    "2 multiFB/d-a/DAM",
    "1 singleFB/App",
    "1 singleFB/DAM",
    "1 singleFB/Corr",
]

feedback_names = [
    "multiFB_c-d-a_App",
    "multiFB_c-d-a_DAM",
    "multiFB_c-d-a_Corr",
    "multiFB_a-c_App",
    "multiFB_a-c_Corr",
    "multiFB_c-a_App",
    "multiFB_c-a_Corr",
    "multiFB_a-d-c_App",
    "multiFB_a-d-c_DAM",
    "multiFB_a-d-c_Corr",
    "multiFB_d-c-a_App",
    "multiFB_d-c-a_DAM",
    "multiFB_d-c-a_Corr",
    "multiFB_d-c_DAM",
    "multiFB_d-c_Corr",
    "multiFB_a-d_App",
    "multiFB_a-d_DAM",
    "multiFB_c-a-d_App",
    "multiFB_c-a-d_DAM",
    "multiFB_c-a-d_Corr",
    "multiFB_d-a-c_App",
    "multiFB_d-a-c_DAM",
    "multiFB_d-a-c_Corr",
    "multiFB_c-d_DAM",
    "multiFB_c-d_Corr",
    "multiFB_a-c-d_App",
    "multiFB_a-c-d_DAM",
    "multiFB_a-c-d_Corr",
    "multiFB_d-a_App",
    "multiFB_d-a_DAM",
    "singleFB_App",
    "singleFB_DAM",
    "singleFB_Corr",
]

csv_paths = get_csv_paths(output_dir)

iteration_data = extract_column(csv_paths, "iterations")

penalty_data = extract_column(csv_paths, "penalty")

iteration_df = pd.DataFrame(iteration_data)
penalty_df = pd.DataFrame(penalty_data)

iteration_df = iteration_df.transpose()
penalty_df = penalty_df.transpose()

penalty_df.columns = feedback_types

result_df = pd.concat([iteration_df, penalty_df], axis=1)

result_df.to_csv("penalty_and_iteration_data_17.csv", index=False)


current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_name = "penalty_and_iteration_data_17.csv"
path = os.path.join(current_directory, dataset_name)

multi_data = extract_entropies_17(path)

datasets, iterations, entropies = zip(*multi_data)

df_multi = pd.DataFrame(
    {"Dataset": datasets, "Iteration": iterations, "Entropy": entropies}
)

df_multi.to_csv("entropy_values_17.csv", index=False)

final_data = manipulate_csv("entropy_values_17.csv")
final_data.to_csv("filtered_entropy_values_17.csv", index=False)

update_iteration_17("filtered_entropy_values_17.csv", "graph_ready_entropy_17.csv")

# Load the dataset
df = pd.read_csv("graph_ready_entropy_17.csv")

dataset_data = {}

# Loop through the unique datasets and populate the dict
unique_datasets = df["Dataset"].unique()
for dataset in unique_datasets:
    dataset_data[dataset] = df[df["Dataset"] == dataset]

color_map = plt.get_cmap("tab20")

plt.figure(figsize=(12, 6))
ax = plt.gca()  # Get the current axis

# Loop through the dictionary and plot the data
for idx, (dataset, data) in enumerate(dataset_data.items()):
    color = color_map(idx / len(dataset_data))
    ax.plot(
        data["Iteration"],
        data["Entropy"],
        marker="o",
        markersize=10,
        label=dataset,
        linewidth=2.5,
        color=color,
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("Entropy")
ax.set_title("Iteration vs. Entropy for Different Datasets")
ax.legend()
ax.grid(True)
plt.tight_layout()

# Check for any missing points
total_points = df.shape[0]
plotted_points = sum(len(data) for data in dataset_data.values())
print("Total points in CSV:", total_points)
print("Plotted points:", plotted_points)
print("Missing points:", total_points - plotted_points)

plt.savefig(os.path.join(output_dir, "iteration_vs_entropy_17.png"))
plt.close()


current_directory = os.path.dirname(os.path.abspath(__file__))
datasets_folder = "datasets/17"
datasets_path = os.path.join(current_directory, datasets_folder)

csv_paths = get_csv_paths(output_dir)  
feedback_names_iter = iter(feedback_names)

csv_paths_17 = []

for path in csv_paths:
    iteration_data = extract_column(path, "iterations")
    x_data = extract_column(path, "x")
    y_data = extract_column(path, "y")
    traffic_data = extract_column(path, "traffic")
    action_data = extract_column(path, "action")
    penalty_data = extract_column(path, "penalty")

    data_dict = {
        "Iterations": [item for sublist in iteration_data for item in sublist],
        "X": [item for sublist in x_data for item in sublist],
        "Y": [item for sublist in y_data for item in sublist],
        "Traffic": [item for sublist in traffic_data for item in sublist],
        "Action": [item for sublist in action_data for item in sublist],
        "Penalty": [item for sublist in penalty_data for item in sublist],
    }

    result_df = pd.DataFrame(data_dict)

    result_df_cleaned = clean_data(result_df)
   
    feedback_name = next(feedback_names_iter)

    output_path = os.path.join(datasets_path, f"{feedback_name}.csv")

    result_df_cleaned.to_csv(output_path, index=False)

    csv_paths_17.append(output_path)


game_txt_paths = [
    testset_filename_one,
    testset_filename_two,
    testset_filename_three,
    testset_filename_four,
]

predict_paths = []

for path in game_txt_paths:
    new_csv_path = convert_txt(path)
    predict_paths.append(new_csv_path)

trained_models = train_17(csv_paths_17)
predict_17(trained_models, predict_paths)

final_data = []

df_entropy = pd.read_csv("graph_ready_entropy_17.csv")

df_penalty = pd.read_csv("penalty_predictions_17.csv")

feedbacks = df_entropy["Dataset"].unique()

training = "game1.csv"
testings = ["game2.csv", "game3.csv", "game4.csv"]

for feedback in feedbacks:
    for testing in testings:
        row_data = {"Feedback": feedback, "Training": training, "Testing": testing}

        # Fetch the entropy value for this feedback type at the 2500th iteration
        entropy_val = df_entropy.loc[
            (df_entropy["Dataset"] == feedback) & (df_entropy["Iteration"] == 2500),
            "Entropy",
        ]
        if not entropy_val.empty:
            row_data["Entropy"] = entropy_val.iloc[0]

        # Extract the last part of the feedback to match with Model
        feedback_last_part = feedback.split("/")[-1]

        # Fetch accuracy and F1 score for this feedback type and testing set
        accuracy_val = df_penalty.loc[
            (df_penalty["Model"].str.contains(feedback_last_part))
            & (df_penalty["Dataset"] == testing),
            "Accuracy",
        ]
        f1_val = df_penalty.loc[
            (df_penalty["Model"].str.contains(feedback_last_part))
            & (df_penalty["Dataset"] == testing),
            "F1_Score",
        ]

        if not accuracy_val.empty and not f1_val.empty:
            row_data["Accuracy"] = accuracy_val.iloc[0]
            row_data["F1_Score"] = f1_val.iloc[0]

        final_data.append(row_data)

final_df = pd.DataFrame(final_data)

final_df.to_csv("Phase_1_final_results_17.csv", index=False)

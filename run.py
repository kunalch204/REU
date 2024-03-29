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
)
from train import clean_data, train, predict, convert_txt, demo_train, demo_predict
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
gridSize = 6

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


# # ---------------------------GENERATE TRAIN SET-------------------------------

# Get oracle policy for trainGrid
_, trainGrid_oracle_policy = valueIteration(trainGrid, is_oracle=True)

# Get single feedback - one of DAM, Approval or Correction
single_fb(trainGrid, trainGrid_oracle_policy, iterations, output_dir)

# Get multiple feedbacks - combination using any 2 of DAM, Approval and Correction or all 3
multiple_fb(trainGrid, trainGrid_oracle_policy, iterations, output_dir)

# # ---------------------------GENERATE TEST SET--------------------------------


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

result_df.to_csv("penalty_and_iteration_data.csv", index=False)

current_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_directory, "penalty_and_iteration_data.csv")

multi_data = multi_extract_entropy(path)

datasets, iterations, entropies = zip(*multi_data)

df_multi = pd.DataFrame(
    {"Dataset": datasets, "Iteration": iterations, "Entropy": entropies}
)

df_multi.to_csv("entropy_values.csv", index=False)

final_data = manipulate_csv("entropy_values.csv")
final_data.to_csv("filtered_entropy_values.csv", index=False)

update_iteration("filtered_entropy_values.csv", "graph_ready_entropy.csv")

# Load the dataset
df = pd.read_csv("graph_ready_entropy.csv")

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

plt.savefig(os.path.join(output_dir, "iteration_vs_entropy.png"))
plt.close()

csv_paths = get_csv_paths(output_dir)

iteration_data = extract_column(csv_paths, "iterations")
traffic_data = extract_column(csv_paths, "traffic")
action_data = extract_column(csv_paths, "action")
penalty_data = extract_column(csv_paths, "penalty")
x_data = extract_column(csv_paths, "x")
y_data = extract_column(csv_paths, "y")

data_dict = {
    "Iterations": [item for sublist in iteration_data for item in sublist],
    "X": [item for sublist in x_data for item in sublist],
    "Y": [item for sublist in y_data for item in sublist],
    "Traffic": [item for sublist in traffic_data for item in sublist],
    "Action": [item for sublist in action_data for item in sublist],
    "Penalty": [item for sublist in penalty_data for item in sublist],
}


result_df = pd.DataFrame(data_dict)

# Clean the data
result_df_cleaned = clean_data(result_df)

result_df_cleaned.to_csv("combined_data.csv", index=False)

num_columns = len(result_df_cleaned.columns)

data_points_per_csv = len(result_df_cleaned) // len(feedback_names)
print(f"DATA Points per a csv: {data_points_per_csv}")

num_csvs = (len(result_df_cleaned) + data_points_per_csv - 1) // data_points_per_csv


datasets_folder = "datasets"
current_directory = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(current_directory, datasets_folder)

csv_paths = []

# Iterate through the range of CSVs and create and save each one
for i in range(num_csvs):
    start_idx = i * data_points_per_csv
    end_idx = (i + 1) * data_points_per_csv
    data_subset = result_df_cleaned.iloc[start_idx:end_idx]
    csv_filename = os.path.join(datasets_path, f"{feedback_names[i]}.csv")
    data_subset.to_csv(csv_filename, index=False)

    csv_paths.append(csv_filename)

game_txt_paths = [
    testset_filename_one,
    testset_filename_two,
    testset_filename_three,
    testset_filename_four,
]

print(len(game_txt_paths))

predict_paths = []

for path in game_txt_paths:
    new_csv_path = convert_txt(path)
    predict_paths.append(new_csv_path)

trained_models = train(csv_paths)
predict(trained_models, predict_paths)

final_data = []

df_entropy = pd.read_csv("graph_ready_entropy.csv")

df_penalty = pd.read_csv("penalty_predictions.csv")

feedbacks = df_entropy["Dataset"].unique()

training = "game1.csv"
testings = ["game2.csv", "game3.csv", "game4.csv"]

for feedback in feedbacks:
    for testing in testings:
        row_data = {"Feedback": feedback, "Training": training, "Testing": testing}

        # Fetch the entropy value for this feedback type at the 200th iteration
        entropy_val = df_entropy.loc[
            (df_entropy["Dataset"] == feedback) & (df_entropy["Iteration"] == 200),
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

final_df.to_csv("Phase_1_final_results_6.csv", index=False)


# PHASE 2 #

trial_names = [
    "55_RZ_2464601_Aug-11-10-18-09",
    "59_RZ_2494228_Aug-11-18-35-07",
    "72_RZ_2903977_Aug-16-12-25-04",
    "79_RZ_3074177_Aug-18-11-46-29",
    "149_JAW_3355334_Dec-15-10-31-51",
    "151_JAW_3358283_Dec-15-11-19-24",
    "156_KM_6306308_Jan-18-14-13-55",
    "157_KM_6307437_Jan-18-14-31-43",
    "616_RZ_5373632_Aug-09-13-01-44",
    "617_RZ_5374717_Aug-09-13-19-21",
]

current_directory = os.path.dirname(os.path.abspath(__file__))

all_paths = []


hf_performance_analysis_folder = "HF-performance-analysis"
atari_freeway_folder = "atari_freeway"
data_folder = "data"
freeway_folder = "freeway"

atari_freeway_path = os.path.abspath(os.path.join(current_directory, "..", hf_performance_analysis_folder, atari_freeway_folder, data_folder, freeway_folder))

# Iterate over each trial name to construct file paths
for trial_name in trial_names:
    base_path = os.path.join(atari_freeway_path, trial_name)
    
    demo_path = os.path.join(base_path, "demo.csv")
    demo_gaze_path = os.path.join(base_path, "demo_gaze.csv")
    
    all_paths.extend([demo_path, demo_gaze_path])

demo_get_entropy(all_paths)

df = pd.read_csv("DEMO_entropy_values.csv")
df = df.sort_values("Dataset Name")
fig, ax = plt.subplots()
labels = df["Dataset Name"]
ind = range(len(labels))
ax.bar(ind, df["Demo Entropy"], width=0.4, label="Demo Entropy", align="edge")
ax.bar(
    ind, df["Demo Gaze Entropy"], width=-0.4, label="Demo Gaze Entropy", align="edge"
)  
ax.set_xlabel("Dataset Name")
ax.set_ylabel("Entropy Value")
ax.set_title("Demo and Demo Gaze Entropy Values")
ax.set_xticks(ind)
ax.set_xticklabels(labels, rotation="vertical")  

# Ensure that the full labels are visible and not cut-off
fig.autofmt_xdate()
ax.legend()
fig.savefig("Demo_and_Demo_Gaze_Entropy_Values.png", bbox_inches="tight")

trained_models = demo_train(all_paths, trial_names)
demo_predict(trained_models, all_paths, trial_names)

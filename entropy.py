import pandas as pd
import os
import numpy as np
import re
import csv
import math


def get_csv_paths(directory):
    csv_paths = []

    for root, dirs, files in os.walk(directory):
        csv_files = [file for file in files if file.endswith(".csv")]
        for file in csv_files:
            file_path = os.path.join(root, file)
            csv_paths.append(file_path)

    return csv_paths


def extract_column(csv_paths, column_name):
    column_data = []

    for path in csv_paths:
        data = pd.read_csv(path, index_col=None, skiprows=1)
        data.columns = ["iterations", "x", "y", "traffic", "action", "penalty"]

        if column_name == "x" or column_name == "y":
            values = (
                data[column_name]
                .str.replace("[^0-9.]", "", regex=True)
                .astype(float)
                .values
            )
        else:
            values = data[column_name].values

        column_data.append(values)

    return column_data


def calculate_entropy(data):
    # Function to calculate the entropy of a given dataset

    # Get the unique values and the counts of each
    unique_values, value_counts = np.unique(data, return_counts=True)

    # Calculate the probabilities of each unique value
    probabilities = value_counts / len(data)  # prob of each value

    # Calculate the entropy using the entropy formula
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Return the calculated entropy
    return entropy


def multi_extract_entropy(dataset_path):
    data = pd.read_csv(dataset_path)
    entropies = []

    num_datasets = 33

    for i in range(0, num_datasets):
        iterations_data = data.iloc[:, i].values
        penalty_col_index = i + num_datasets
        penalty_data = data.iloc[:, penalty_col_index].values

        combined_data = list(zip(iterations_data, penalty_data))

        # Group the data based on unique iteration values
        grouped_data = {}
        for it, pen in combined_data:
            if it not in grouped_data:
                grouped_data[it] = []
            grouped_data[it].append(pen)

        # Extract dataset name and number of datasets for the group from the header
        header = data.columns[i + num_datasets]
        dataset_name_match = re.match(r"\d+ multiFB/\w+/(\w+)", header)
        if dataset_name_match:
            dataset_name = dataset_name_match.group(1)
            num_datasets_for_folder = int(dataset_name_match.group(0).split()[0])
        else:
            dataset_name = header
            num_datasets_for_folder = 1
        # Calculate entropy for each group and append to entropies list
        for it, penalties in grouped_data.items():
            entropy = calculate_entropy(penalties)
            for _ in range(num_datasets_for_folder):
                entropies.append((dataset_name, it, entropy))
    return entropies


def manipulate_csv(csv_file_name):
    df = pd.read_csv(csv_file_name)

    data = df["Dataset"]

    Set = []  # the set type ex: 3 multiFB/c-d-a
    Feedback_Type = []  # the feedback type ex: App, Corr, DAM

    # Iterate through the dataset columns and grab there set and feedback type
    for entry in data:
        parts = entry.split(" ")
        dataset_name = parts[0] + " " + parts[1][: parts[1].rfind("/")]
        dataset_feedback = " ".join(parts[1:])[parts[1].rfind("/") + 1 :]
        Set.append(dataset_name)
        Feedback_Type.append(dataset_feedback)

    # Create dict containing the feedback groups and their 'final say'
    group_feedback_types = {
        "3 multiFB/c-d-a": "App",
        "2 multiFB/a-c": "Corr",
        "2 multiFB/c-a": "App",
        "3 multiFB/a-d-c": "Corr",
        "3 multiFB/d-c-a": "App",
        "2 multiFB/d-c": "Corr",
        "2 multiFB/a-d": "DAM",
        "3 multiFB/c-a-d": "DAM",
        "3 multiFB/d-a-c": "Corr",
        "2 multiFB/c-d": "DAM",
        "3 multiFB/a-c-d": "DAM",
        "2 multiFB/d-a": "App",
    }

    # Iterate through the extracted feedback types and filter rows
    rows_to_keep = []
    for idx, feedback_type in enumerate(Feedback_Type):
        dataset_key = Set[idx]
        expected_feedback_type = group_feedback_types.get(dataset_key)

        if (
            expected_feedback_type is not None
            and feedback_type == expected_feedback_type
        ):
            rows_to_keep.append(idx)
        elif dataset_key.startswith("1"):
            rows_to_keep.append(idx)

    # Filter the DataFrame based on the rows to keep
    df_filtered = df.iloc[rows_to_keep]

    return df_filtered


def update_iteration(input_csv_file_name, output_csv_file_name):
    df = pd.read_csv(input_csv_file_name)

    for i in range(len(df)):
        dataset = str(df.loc[i, "Dataset"])
        iteration = int(df.loc[i, "Iteration"])

        if dataset.startswith("3"):
            df.loc[i, "Iteration"] = iteration * 3
        elif dataset.startswith("2"):
            df.loc[i, "Iteration"] = iteration * 2

        parts = dataset.split("/")
        if dataset.startswith("1"):
            df.loc[i, "Dataset"] = "/" + "/".join(parts[-1:])
        if len(parts) > 2:
            df.loc[i, "Dataset"] = "/" + "/".join(parts[-2:])

    i = 0
    while i < len(df):
        dataset = str(df.loc[i, "Dataset"])
        iteration = int(df.loc[i, "Iteration"])

        if i > 0:
            prev_dataset = str(df.loc[i - 1, "Dataset"])
            prev_iteration = int(df.loc[i - 1, "Iteration"])

            # case 1
            if dataset == prev_dataset and dataset.endswith("App"):
                if iteration == 128 and prev_iteration == 80:
                    df.loc[i, "Iteration"] = 160
                    new_row = df.loc[i].copy()
                    new_row["Iteration"] = 200
                    df = pd.concat(
                        [df.iloc[: i + 1], pd.DataFrame([new_row]), df.iloc[i + 1 :]]
                    ).reset_index(drop=True)
                    i += 1  # Skip the newly added row

            #  case 2
            if dataset.endswith("App") and iteration == 64 and prev_iteration == 40:
                df.loc[i, "Iteration"] = 80
                new_row_1 = df.loc[i].copy()
                new_row_1["Iteration"] = 160
                new_row_2 = df.loc[i].copy()
                new_row_2["Iteration"] = 200
                df = pd.concat(
                    [
                        df.iloc[: i + 1],
                        pd.DataFrame([new_row_1, new_row_2]),
                        df.iloc[i + 1 :],
                    ]
                ).reset_index(drop=True)
                i += 2  # Skip the newly added rows

        i += 1

    # Round up everything to the nearest tens, with an exception for 192
    for i in range(len(df)):
        iteration = int(df.loc[i, "Iteration"])
        if iteration == 192:
            df.loc[i, "Iteration"] = 200
        else:
            df.loc[i, "Iteration"] = int(math.ceil(iteration / 10.0)) * 10

    df.to_csv(output_csv_file_name, index=False)


# PHASE 2 #


def demo_get_entropy(csv_paths):
    with open("DEMO_entropy_values.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Dataset Name", "Demo Entropy", "Demo Gaze Entropy"])
        for i in range(0, len(csv_paths), 2):
            demo_path = csv_paths[i]
            demo_gaze_path = csv_paths[i + 1]
            # Calculate entropy for demo
            demo_data = pd.read_csv(demo_path)
            demo_penalty_data = demo_data.iloc[:, -1].values
            demo_entropy = calculate_entropy(demo_penalty_data)
            # Calculate entropy for demo gaze
            demo_gaze_data = pd.read_csv(demo_gaze_path)
            demo_gaze_penalty_data = demo_gaze_data.iloc[:, -1].values
            demo_gaze_entropy = calculate_entropy(demo_gaze_penalty_data)
            dataset_name = os.path.basename(os.path.dirname(demo_path))
            csvwriter.writerow([dataset_name, demo_entropy, demo_gaze_entropy])


# Gridsize 17 #


def extract_entropies_17(dataset_path):
    data = pd.read_csv(dataset_path)
    entropies = []

    num_datasets = 33

    for i in range(num_datasets):
        iterations_data = data.iloc[:, i].dropna().values
        penalty_col_index = i + num_datasets
        penalty_data = data.iloc[:, penalty_col_index].dropna().values

        combined_data = list(zip(iterations_data, penalty_data))

        grouped_data = {}
        for it, pen in combined_data:
            if not np.isnan(it):
                if it not in grouped_data:
                    grouped_data[it] = []
                grouped_data[it].append(pen)

        header = data.columns[i + num_datasets]
        dataset_name_match = re.match(r"(\d+ multiFB/[\w-]+/[\w-]+)", header)
        if dataset_name_match:
            dataset_name = dataset_name_match.group(1)
        else:
            dataset_name = header

        for it, penalties in grouped_data.items():
            if penalties:  # Ensure the penalty group is not empty
                entropy = calculate_entropy(penalties)
                entropies.append((dataset_name, it, entropy))
    return entropies


def update_iteration_17(input_csv_file_name, output_csv_file_name):
    df = pd.read_csv(input_csv_file_name)

    for i in range(len(df)):
        dataset = str(df.loc[i, "Dataset"])
        iteration = int(df.loc[i, "Iteration"])

        if dataset.startswith("3"):
            df.loc[i, "Iteration"] = iteration * 3
        elif dataset.startswith("2"):
            df.loc[i, "Iteration"] = iteration * 2

        parts = dataset.split("/")
        if dataset.startswith("1"):
            df.loc[i, "Dataset"] = "/" + "/".join(parts[-1:])
        if len(parts) > 2:
            df.loc[i, "Dataset"] = "/" + "/".join(parts[-2:])

    i = 0
    while i < len(df):
        dataset = str(df.loc[i, "Dataset"])
        iteration = int(df.loc[i, "Iteration"])

        if i > 0:
            prev_dataset = str(df.loc[i - 1, "Dataset"])
            prev_iteration = int(df.loc[i - 1, "Iteration"])

            # Special conditions 
            if dataset == prev_dataset and dataset.endswith("App"):
                if iteration == 1800 and prev_iteration == 1600:
                    df.loc[i, "Iteration"] = 2000
                    new_row = df.loc[i].copy()
                    new_row["Iteration"] = 2500
                    new_row["Entropy"] = df.loc[i, "Entropy"]
                    df = pd.concat(
                        [df.iloc[: i + 1], pd.DataFrame([new_row]), df.iloc[i + 1 :]]
                    ).reset_index(drop=True)
                    i += 1  # Skip the newly added row

                elif iteration == 900 and prev_iteration == 800:
                    df.loc[i, "Iteration"] = 1200
                    entropy_value = df.loc[i, "Entropy"]
                    new_rows = [
                        {
                            "Dataset": dataset,
                            "Iteration": 1600,
                            "Entropy": entropy_value,
                        },
                        {
                            "Dataset": dataset,
                            "Iteration": 2000,
                            "Entropy": entropy_value,
                        },
                        {
                            "Dataset": dataset,
                            "Iteration": 2500,
                            "Entropy": entropy_value,
                        },
                    ]
                    df = pd.concat(
                        [df.iloc[: i + 1], pd.DataFrame(new_rows), df.iloc[i + 1 :]]
                    ).reset_index(drop=True)
                    i += 3  # Skip the newly added rows

        i += 1

    # Round every iteration value to the nearest 100
    df["Iteration"] = (df["Iteration"] + 50) // 100 * 100

    df.to_csv(output_csv_file_name, index=False)

import pandas as pd
import os
import numpy as np
import re
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
)
from sklearn.metrics import f1_score, accuracy_score
import pickle
from sklearn.metrics import confusion_matrix, classification_report


def clean_data(df):

    # Replace "False" with 0 and "True" with 1 in the "Traffic" column
    df["Traffic"] = df["Traffic"].replace({" False)": 0, " True)": 1})

    # Remove extra ")" characters and convert the "Action" column values to integers
    df["Action"] = df["Action"].str.replace(")", "").astype(int)

    return df


def train(csv_paths):
    results = []
    trained_models = []

    for path in csv_paths:
        # Load dataset
        dataset = pd.read_csv(path)
        print(path)

        # Extract features and target
        X = dataset[["X", "Y", "Traffic", "Action"]]
        y = dataset["Penalty"]

        # Number of trees in the random forest
        n_estimators = [int(x) for x in np.linspace(start=30, stop=500, num=5)]
        # Number of features considered for each split
        max_features = ["sqrt"]
        # Maximum depth of individual decision trees
        max_depth = np.arange(2, 5)
        # Minimum number of samples required to split an internal node
        min_samples_split = [2]
        # Minimum number of samples required to be at a leaf node
        min_samples_leaf = [1]
        # Whether samples are drawn with or without replacement
        bootstrap = [True]

        # Create parameter distributions for RandomizedSearchCV
        param_dist = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }

        rf = RandomForestClassifier()

        # RandomizedSearchCV for initial hyperparameter search
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=2,
            n_jobs=4,
            random_state=42,
        )
        random_search.fit(X, y)

        # Get best parameters from RandomizedSearchCV
        best_params_random = random_search.best_params_

        # Create parameter grid for GridSearchCV around best parameters
        param_grid = {
            "n_estimators": [best_params_random["n_estimators"]],
            "max_features": [best_params_random["max_features"]],
            "max_depth": np.arange(
                best_params_random["max_depth"] - 1, best_params_random["max_depth"] + 2
            ),
            "min_samples_split": [best_params_random["min_samples_split"]],
            "min_samples_leaf": [best_params_random["min_samples_leaf"]],
            "bootstrap": [best_params_random["bootstrap"]],
        }

        # GridSearchCV for fine-tuning with narrowed parameter ranges
        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=4
        )
        grid_search.fit(X, y)

        # Store best model and dataset name
        trained_models.append((grid_search.best_estimator_, os.path.basename(path)))

        # Store results for analysis
        results.append(
            {
                "Dataset": os.path.basename(path),
                "Best_Params": grid_search.best_params_,
                "Avg_CV_Score": grid_search.best_score_,
                "CV_Std_Dev": grid_search.cv_results_["std_test_score"][
                    grid_search.best_index_
                ],
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv("model_results.csv", index=False)

    return trained_models


def train_17(csv_paths):
    results = []
    trained_models = []

    for path in csv_paths:
        # Load dataset
        dataset = pd.read_csv(path)
        print(path)

        # Extract features and target
        X = dataset[["X", "Y", "Traffic", "Action"]]
        y = dataset["Penalty"]

        # Number of trees in the random forest
        n_estimators = [int(x) for x in np.linspace(start=30, stop=500, num=5)]
        # Number of features considered for each split
        max_features = ["sqrt"]
        # Maximum depth of individual decision trees
        max_depth = np.arange(2, 5)
        # Minimum number of samples required to split an internal node
        min_samples_split = [2]
        # Minimum number of samples required to be at a leaf node
        min_samples_leaf = [1]
        # Whether samples are drawn with or without replacement
        bootstrap = [True]

        # Create parameter distributions for RandomizedSearchCV
        param_dist = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }

        rf = RandomForestClassifier()

        # RandomizedSearchCV for initial hyperparameter search
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=2,
            n_jobs=4,
            random_state=42,
        )
        random_search.fit(X, y)

        # Get best parameters from RandomizedSearchCV
        best_params_random = random_search.best_params_

        # Create parameter grid for GridSearchCV around best parameters
        param_grid = {
            "n_estimators": [best_params_random["n_estimators"]],
            "max_features": [best_params_random["max_features"]],
            "max_depth": np.arange(
                best_params_random["max_depth"] - 1, best_params_random["max_depth"] + 2
            ),
            "min_samples_split": [best_params_random["min_samples_split"]],
            "min_samples_leaf": [best_params_random["min_samples_leaf"]],
            "bootstrap": [best_params_random["bootstrap"]],
        }

        # GridSearchCV for fine-tuning with narrowed parameter ranges
        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=4
        )
        grid_search.fit(X, y)

        # Store best model and dataset name
        trained_models.append((grid_search.best_estimator_, os.path.basename(path)))

        # Store results for analysis
        results.append(
            {
                "Dataset": os.path.basename(path),
                "Best_Params": grid_search.best_params_,
                "Avg_CV_Score": grid_search.best_score_,
                "CV_Std_Dev": grid_search.cv_results_["std_test_score"][
                    grid_search.best_index_
                ],
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv("model_results_17.csv", index=False)

    return trained_models


def predict(trained_models, predict_paths):
    predictions = []

    for model, trained_dataset in trained_models:
        model_predictions = []
        model_accuracies = []
        model_f1_scores = []

        for path in predict_paths:
            predict_dataset = pd.read_csv(path)
            X_predict = predict_dataset[["X", "Y", "Traffic", "Action"]]
            y_actual = predict_dataset["Penalty"]

            y_pred = model.predict(X_predict)
            accuracy = accuracy_score(y_actual, y_pred)
            f1 = f1_score(y_actual, y_pred, average="weighted")

            model_predictions.append(y_pred)
            model_accuracies.append(accuracy)
            model_f1_scores.append(f1)

        predictions.append(
            {
                "Model": model,
                "Trained_Dataset": trained_dataset,
                "Predictions": model_predictions,
                "Accuracy": model_accuracies,
                "F1_Score": model_f1_scores,
            }
        )

    result_data = []
    for model_data in predictions:
        model = model_data["Model"]
        trained_dataset = model_data["Trained_Dataset"]
        for i, path in enumerate(predict_paths):
            dataset_name = os.path.basename(path)
            predictions = model_data["Predictions"][i]
            accuracy = model_data["Accuracy"][i]
            f1 = model_data["F1_Score"][i]

            result_data.append(
                {
                    "Model": f"Trained on {trained_dataset}",
                    "Dataset": dataset_name,
                    "Predictions": predictions,
                    "Accuracy": accuracy,
                    "F1_Score": f1,
                }
            )

    result_df = pd.DataFrame(result_data)
    result_df.to_csv("penalty_predictions.csv", index=False)


def predict_17(trained_models, predict_paths):
    predictions = []

    for model, trained_dataset in trained_models:
        model_predictions = []
        model_accuracies = []
        model_f1_scores = []

        for path in predict_paths:
            predict_dataset = pd.read_csv(path)
            X_predict = predict_dataset[["X", "Y", "Traffic", "Action"]]
            y_actual = predict_dataset["Penalty"]

            y_pred = model.predict(X_predict)
            accuracy = accuracy_score(y_actual, y_pred)
            f1 = f1_score(y_actual, y_pred, average="weighted")

            model_predictions.append(y_pred)
            model_accuracies.append(accuracy)
            model_f1_scores.append(f1)

        predictions.append(
            {
                "Model": model,
                "Trained_Dataset": trained_dataset,
                "Predictions": model_predictions,
                "Accuracy": model_accuracies,
                "F1_Score": model_f1_scores,
            }
        )

    result_data = []
    for model_data in predictions:
        model = model_data["Model"]
        trained_dataset = model_data["Trained_Dataset"]
        for i, path in enumerate(predict_paths):
            dataset_name = os.path.basename(path)
            predictions = model_data["Predictions"][i]
            accuracy = model_data["Accuracy"][i]
            f1 = model_data["F1_Score"][i]

            result_data.append(
                {
                    "Model": f"Trained on {trained_dataset}",
                    "Dataset": dataset_name,
                    "Predictions": predictions,
                    "Accuracy": accuracy,
                    "F1_Score": f1,
                }
            )

    result_df = pd.DataFrame(result_data)
    result_df.to_csv("penalty_predictions_17.csv", index=False)


def convert_txt(path):
    folder_name = "testSets"

    with open(path, "r") as txtfile:
        txtfile.readline()
        csv_filename = os.path.basename(path).replace(".txt", ".csv")
        csv_path = os.path.join(folder_name, csv_filename)

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["X", "Y", "Traffic", "Action", "Penalty"])

            for line in txtfile:
                match = re.match(
                    r"\(\(\((\d+), (\d+)\), (True|False)\),(\d+),(\d+)\)", line.strip()
                )
                if match:
                    x, y, traffic, action, penalty = match.groups()
                    traffic = "1" if traffic == "True" else "0"
                    csv_writer.writerow([x, y, traffic, action, penalty])

    return csv_path


def demo_train(csv_paths, trial_names):
    results = []
    trained_models = []

    for i, path in enumerate(csv_paths):
        dataset = pd.read_csv(path, header=None)

        trial_name = trial_names[i // 2]
        file_type = "demo" if "demo.csv" in path else "demo_gaze"
        dataset_name = f"{trial_name}_{file_type}"

        print(f"Training on dataset {dataset_name}")

        # Extract features and target
        X = dataset.iloc[:, :-2]
        y = dataset.iloc[:, -1]

        rf = RandomForestClassifier()

        # Create parameter distributions for RandomizedSearchCV
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start=30, stop=500, num=4)],
            "max_features": ["sqrt"],
            "max_depth": np.arange(2, 5),
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "bootstrap": [True],
        }

        # RandomizedSearchCV for initial hyperparameter search
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=2,
            n_jobs=4,
            random_state=42,
        )
        random_search.fit(X, y)

        # Get best parameters from RandomizedSearchCV
        best_params_random = random_search.best_params_

        # Create parameter grid for GridSearchCV around best parameters
        param_grid = {
            "n_estimators": [best_params_random["n_estimators"]],
            "max_features": [best_params_random["max_features"]],
            "max_depth": np.arange(
                best_params_random["max_depth"] - 1, best_params_random["max_depth"] + 2
            ),
            "min_samples_split": [best_params_random["min_samples_split"]],
            "min_samples_leaf": [best_params_random["min_samples_leaf"]],
            "bootstrap": [best_params_random["bootstrap"]],
        }

        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=4
        )
        grid_search.fit(X, y)

        # Store best model from GridSearchCV and the dataset it was trained on
        trained_models.append(
            {"model": grid_search.best_estimator_, "trained_on": dataset_name}
        )

        # Store results for analysis
        results.append(
            {
                "Dataset": dataset_name,
                "Best_Params": grid_search.best_params_,
                "Avg_CV_Score": grid_search.best_score_,
                "CV_Std_Dev": grid_search.cv_results_["std_test_score"][
                    grid_search.best_index_
                ],
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv("DEMO_model_results.csv", index=False)

    return trained_models


def demo_predict(trained_models, all_paths, trial_names):
    result_data = []
    demo_accuracies = []
    demo_gaze_accuracies = []
    demo_f1_scores = []
    demo_gaze_f1_scores = []
    for model_dict in trained_models:
        model_name = f"Model trained on {model_dict['trained_on']}"
        model = model_dict["model"]
        for i, test_path in enumerate(all_paths):
            test_dataset = pd.read_csv(test_path, header=None)
            X_test = test_dataset.iloc[:, :-2]
            y_test = test_dataset.iloc[:, -1]
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            trial_name = trial_names[i // 2]
            file_type = "demo" if "demo.csv" in test_path else "demo_gaze"
            dataset_name = f"{trial_name}_{file_type}"
            result_data.append(
                {
                    "Model": model_name,
                    "Tested_On_Dataset": dataset_name,
                    "Accuracy": accuracy,
                    "F1_Score": f1,
                }
            )
            if file_type == "demo":
                demo_accuracies.append(accuracy)
                demo_f1_scores.append(f1)
            else:
                demo_gaze_accuracies.append(accuracy)
                demo_gaze_f1_scores.append(f1)

    avg_demo_accuracy = np.mean(demo_accuracies)
    std_demo_accuracy = np.std(demo_accuracies)
    avg_demo_gaze_accuracy = np.mean(demo_gaze_accuracies)
    std_demo_gaze_accuracy = np.std(demo_gaze_accuracies)
    avg_demo_f1 = np.mean(demo_f1_scores)
    std_demo_f1 = np.std(demo_f1_scores)
    avg_demo_gaze_f1 = np.mean(demo_gaze_f1_scores)
    std_demo_gaze_f1 = np.std(demo_gaze_f1_scores)

    summary_data = {
        "Summary_Stat": ["Average", "Standard Deviation"],
        "Demo_Accuracy": [avg_demo_accuracy, std_demo_accuracy],
        "Demo_Gaze_Accuracy": [avg_demo_gaze_accuracy, std_demo_gaze_accuracy],
        "Demo_F1": [avg_demo_f1, std_demo_f1],
        "Demo_Gaze_F1": [avg_demo_gaze_f1, std_demo_gaze_f1],
    }

    result_df = pd.DataFrame(result_data)
    summary_df = pd.DataFrame(summary_data)

    result_df.to_csv("DEMO_dataset_predictions_results.csv", index=False)
    summary_df.to_csv("DEMO_dataset_predictions_summary.csv", index=False)

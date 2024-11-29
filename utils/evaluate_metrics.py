import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from flask_sqlalchemy import SQLAlchemy
from models import AuthenticationEvaluation  # Import your model
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI, file-only output
import matplotlib.pyplot as plt
y_true =[3]
def calculate_frr_far(y_true, y_pred):
    # Check if confusion matrix can be computed properly
    if len(set(y_true)) > 1 or len(set(y_pred)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Rejection Rate (FRR)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate (FAR)
    else:
        # If there's no variation in predictions or ground truth, return default values
        frr = 0.0
        far = 0.0

    return frr, far



def evaluate_model_for_all_users():
    # Fetch all evaluation data from the AuthenticationEvaluation table using the existing db session
    results = AuthenticationEvaluation.query.all()

    # Initialize lists for true labels and predicted labels for each model
    y_true = []
    y_pred_log_reg = []
    y_pred_rf = []
    y_pred_svm = []
    y_pred_gb = []
    y_pred_nn = []
    y_pred_log_reg_username = []
    y_pred_rf_username = []
    y_pred_svm_username = []
    y_pred_gb_username = []
    y_pred_nn_username = []

    # Extract the true labels and model predictions from the fetched results
    for result in results:
        y_true.append(result.ground_truth)

        # Append predictions for each model
        y_pred_log_reg.append(result.logistic_regression_prediction)
        y_pred_rf.append(result.random_forest_prediction)
        y_pred_svm.append(result.support_vector_machine_prediction)
        y_pred_gb.append(result.gradient_boosting_prediction)
        y_pred_nn.append(result.neural_network_prediction)
        y_pred_log_reg_username.append(result.logistic_regression_with_username_prediction)
        y_pred_rf_username.append(result.random_forest_with_username_prediction)
        y_pred_svm_username.append(result.support_vector_machine_with_username_prediction)
        y_pred_gb_username.append(result.gradient_boosting_with_username_prediction)
        y_pred_nn_username.append(result.neural_network_with_username_prediction)

    # Calculate metrics for each model
    metrics = {}

    # Define a helper function to calculate the metrics for each model
    def calculate_model_metrics(y_true, y_pred, model_name):
        precision = precision_score(y_true, y_pred, pos_label='valid', zero_division=1)
        recall = recall_score(y_true, y_pred, pos_label='valid', zero_division=1)
        f1 = f1_score(y_true, y_pred, pos_label='valid', zero_division=1)
        frr, far = calculate_frr_far(y_true, y_pred)
        
        metrics[model_name] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "FRR": frr,
            "FAR": far
        }

    # Calculate metrics for each model
    calculate_model_metrics(y_true, y_pred_log_reg, "Logistic Regression")
    calculate_model_metrics(y_true, y_pred_rf, "Random Forest")
    calculate_model_metrics(y_true, y_pred_svm, "Support Vector Machine")
    calculate_model_metrics(y_true, y_pred_gb, "Gradient Boosting")
    calculate_model_metrics(y_true, y_pred_nn, "Neural Network")
    calculate_model_metrics(y_true, y_pred_log_reg_username, "Logistic Regression with Username")
    calculate_model_metrics(y_true, y_pred_rf_username, "Random Forest with Username")
    calculate_model_metrics(y_true, y_pred_svm_username, "Support Vector Machine with Username")
    calculate_model_metrics(y_true, y_pred_gb_username, "Gradient Boosting with Username")
    calculate_model_metrics(y_true, y_pred_nn_username, "Neural Network with Username")

    return metrics
def plot_metrics(metrics, save_path="metrics_plots"):
    # Ensure the directory exists
    import os
    os.makedirs(save_path, exist_ok=True)

    # Lists for metrics to plot
    metric_names = ["Precision", "Recall", "F1-Score", "FRR", "FAR"]
    
    for metric_name in metric_names:
        # Extract metric values for each model
        models = list(metrics.keys())
        values = [metrics[model][metric_name] for model in models]

        # Plotting the metrics for each model
        plt.figure(figsize=(10, 6))
        plt.barh(models, values, color='skyblue')
        plt.xlabel(metric_name)
        plt.title(f"{metric_name} for Each Model")
        
        # Save each plot to a file
        plt.tight_layout()
        plt.savefig(f"{save_path}/{metric_name.lower()}_plot.png")
        plt.close()  # Close plot to avoid overlap

    print(f"Plots saved to {save_path} folder")

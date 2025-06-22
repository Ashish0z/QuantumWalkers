import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from datasets.synthetic_generator import generate_realistic_sonar_data
from utils.logging_utils import log_config, log_metrics, log_time
from utils.pca_utils import reduce_features
from models.classical.dnn import train_dnn
from models.classical.svm import train_svm
from models.quantum.qnn import train_qnn
from models.quantum.q_svm import train_q_svm

#Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_metrics(metrics, train_size):
    labels = list(metrics.keys())
    acc = [metrics[m]['accuracy'] for m in labels]
    f1 = [metrics[m]['f1'] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/train_{train_size}_metrics.png")
    plt.close()

def run_experiment(train_size, config):

    start_all = time.perf_counter()
    X, y = generate_realistic_sonar_data(n_samples=train_size + 100, seed=config['seed'])
    # Split the dataset into training and testing sets
    X_train, X_test = X[:train_size], X[train_size:]
    # Reduce features if specified in the config
    if config.get('reduce_features', False):
        X_train, X_test, pca = reduce_features(X_train, X_test, n_components=config['n_components'])
    y_train, y_test = y[:train_size], y[train_size:]

    metrics = {}
    print(f"Training size: {train_size}, Test size: {len(y_test)}")
    # SVM
    print("Training SVM...")
    start = time.perf_counter()
    svm_model, svm_preds = train_svm(X_train, y_train, X_test, config=config['models']['svm'])
    t = time.perf_counter() - start
    metrics['SVM'] = {
        'accuracy': accuracy_score(y_test, svm_preds),
        'f1': f1_score(y_test, svm_preds, average='weighted'),
        'confusion': confusion_matrix(y_test, svm_preds).tolist(),
        'report': classification_report(y_test, svm_preds, output_dict=True),
        'time': t
    }
    print("SVM training completed.")
    print(f"SVM Accuracy: {metrics['SVM']['accuracy']:.4f}, F1 Score: {metrics['SVM']['f1']:.4f}")
    
    print("Training DNN...")
    # DNN
    start = time.time()
    dnn_model, dnn_preds = train_dnn(X_train, y_train, X_test, config['models']['dnn'])
    t = time.time() - start
    metrics['DNN'] = {
        'accuracy': accuracy_score(y_test, dnn_preds),
        'f1': f1_score(y_test, dnn_preds, average='weighted'),
        'confusion': confusion_matrix(y_test, dnn_preds).tolist(),
        'report': classification_report(y_test, dnn_preds, output_dict=True),
        'time': t
    }
    print("DNN training completed.")
    print(f"DNN Accuracy: {metrics['DNN']['accuracy']:.4f}, F1 Score: {metrics['DNN']['f1']:.4f}")

    print("Training QNN...")
    # QNN
    start = time.time()
    qnn_model, qnn_preds = train_qnn(X_train, y_train, X_test, config['models']['qnn'])
    t = time.time() - start
    metrics['QNN'] = {
        'accuracy': accuracy_score(y_test, qnn_preds),
        'f1': f1_score(y_test, qnn_preds, average='weighted'),
        'confusion': confusion_matrix(y_test, qnn_preds).tolist(),
        'report': classification_report(y_test, qnn_preds, output_dict=True),
        'time': t
    }
    print("QNN training completed.")
    print(f"QNN Accuracy: {metrics['QNN']['accuracy']:.4f}, F1 Score: {metrics['QNN']['f1']:.4f}")

    print("Training Q-SVM...")
    # Q-SVM
    start = time.time()
    qsvm_model, qsvm_preds = train_q_svm(X_train, y_train, X_test)
    t = time.time() - start
    metrics['Q-SVM'] = {
        'accuracy': accuracy_score(y_test, qsvm_preds),
        'f1': f1_score(y_test, qsvm_preds, average='weighted'),
        'confusion': confusion_matrix(y_test, qsvm_preds).tolist(),
        'report': classification_report(y_test, qsvm_preds, output_dict=True),
        'time': t
    }
    print("Q-SVM training completed.")
    print(f"Q-SVM Accuracy: {metrics['Q-SVM']['accuracy']:.4f}, F1 Score: {metrics['Q-SVM']['f1']:.4f}")

    total_time = time.time() - start_all

    log_metrics(metrics, RESULTS_DIR)
    log_time(total_time, RESULTS_DIR)
    log_config(config, RESULTS_DIR)
    plot_metrics(metrics, train_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_size = config.get('train_size', 100)  # Default to 100 if not specified
    run_experiment(train_size, config)
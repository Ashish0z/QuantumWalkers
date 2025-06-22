# QuantumWalkers 🧠⚛️

**QuantumWalkers** is a benchmarking suite that compares classical and quantum machine learning models—including DNN, SVM, Quantum SVM (Q-SVM), and Quantum Neural Network (QNN)—on synthetic sonar signal classification tasks.

## 📁 Project Structure
project_root/
├── automate.py # Batch experiment runner using AutoMan
├── config.yaml # Configuration file for models and experiments
├── datasets/
│ └── synthetic_generator.py # Synthetic sonar signal data generator
├── experiments/
│ └── exp1.py # Main script for running one experiment
├── models/
│ ├── classical/
│ │ ├── dnn.py # Deep Neural Network
│ │ └── svm.py # Classical SVM
│ └── quantum/
│ ├── q_svm.py # Quantum SVM using precomputed kernel
│ └── qnn.py # Quantum Neural Network using PennyLane
├── results/ # Auto-generated folder to store logs and plots
├── utils/
│ ├── logging_utils.py # Timer and logger helpers
│ └── pca_utils.py # Dimensionality reduction helper for QML

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.9+
- Create a virtual environment:
  ```bash
  python -m venv env
  source env/bin/activate  # or env\Scripts\activate on Windows
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### ⚙️ Running a Single Experiment
You can run the main experiment script:

```bash
python experiments/exp1.py
```

This will:

- Generate synthetic sonar data
- Train DNN, SVM, QNN, and Q-SVM
- Log training time, validation time, accuracy, precision, recall, F1-score
- Save performance plots and logs to the results/ directory


### 🏃‍♂️ Running Batch Experiments

To run multiple experiments in parallel with varying configurations (e.g. dataset size, number of qubits), run:

```bash
python automate.py
```

### 🔧 Configuration
All model parameters and experiment settings are defined in config.yaml.

Example (config.yaml):
```yaml
train_size: 50
seed: 42
reduce_features: True
n_components: 4
models:
  dnn:
    hidden_size: 32
    epochs: 100
    learning_rate: 0.01
  svm:
    C: 0.1
  qnn:
    n_qubits: 4
    entangling_layers: 3
    epochs: 30
    learning_rate: 0.01
  qsvm:
    n_qubits: 4
```
Change n_qubits to match PCA-reduced input dimensions if needed.

### 📊 Output & Logs
All logs and plots are saved under the results/ folder:

- Training/validation time per model
- Accuracy, precision, recall, F1-score
- Confusion matrices (if enabled)
- Plots comparing model performance over different settings

## 🧠 Notes
Q-SVM uses a custom quantum kernel based on AngleEmbedding

Dimensionality reduction (via PCA) is applied to match qubit limits for quantum models

PennyLane is used for QML backends

Torch integration with default.qubit.torch for GPU acceleration (if available)

## 🛠️ TODO
Add real-world dataset support

Extend quantum kernel options (ZZEmbedding, custom ansätze)

Add parameter sweep via grid search

## 👨‍💻 Author
Developed by 
- Ashish Patel
- Sumanth Kotikalapudi
- bhavya sunkari
- Tomoya Hatanaka
- Kishore Nagarajan
- Vacha Buch

 as part of the ISIT Quantum Hackathon project.
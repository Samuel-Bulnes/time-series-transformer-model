Time Series Classification with Transformer Model
This project implements a Transformer-based neural network for time series classification using the FordA dataset from the UCR Time Series Repository.
Pipeline Overview
Time Series Data → Preprocessing → Transformer Encoder → Classification → Prediction
The system processes univariate time series sequences and outputs:

Binary classification (Class 0 or Class 1)
Model accuracy: ~83% on validation set
Training history visualization
Model performance metrics


Environment Setup
Prerequisites

Python 3.8+
R and RStudio (for running the Quarto notebook)
Virtual environment recommended

Installation

Clone the repository:

bashgit clone https://github.com/Samuel-Bulnes/time-series-transformer-model.git
cd time-series-transformer-model

Create and activate virtual environment:

bashpython -m venv ai-env
ai-env\Scripts\activate  # Windows
# source ai-env/bin/activate  # Linux/Mac

Install dependencies:

bashpip install -r requirements.txt
Required Libraries

numpy >= 1.21.0
tensorflow >= 2.8.0
keras >= 2.8.0
matplotlib >= 3.5.0
scikit-learn >= 1.0.0


Running the Project
In RStudio

Open notebooks/time_series_classification.qmd
Configure your virtual environment path:

rlibrary(reticulate)
use_virtualenv("C:/ai-env", required = TRUE)
py_config()

Run code chunks sequentially

Dataset
FordA from UCR Time Series Repository

Univariate sequences of 500 timesteps
Binary classification task
Auto-downloaded from: https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/


Model Architecture
Transformer Encoder

Input Layer: (500, 1) normalized sequences
Transformer Blocks (4 layers):

Multi-Head Attention (4 heads, 256 dimensions)
Layer Normalization
Feed-Forward Network (Conv1D)
Dropout (0.25)
Residual Connections


Global Average Pooling: Temporal aggregation
MLP Head: Dense layer (128 units) + Dropout (0.4)
Output: Softmax classification

Hyperparameters
pythonhead_size = 256
num_heads = 4
ff_dim = 4
num_transformer_blocks = 4
mlp_units = [128]
dropout = 0.25
mlp_dropout = 0.4
learning_rate = 1e-4
batch_size = 64
epochs = 150 (with early stopping)

Results

Training Accuracy: ~85%
Validation Accuracy: ~83%
Test Accuracy: ~82%

The model successfully learns temporal patterns using self-attention mechanisms, demonstrating that Transformers are viable for sequential data beyond NLP tasks.

Project Structure
time-series-transformer-model/
│
├── notebooks/
│   └── time_series_classification.qmd    # Main Quarto notebook
│
├── data/                                   # Dataset (auto-downloaded)
├── results/                                # Training outputs
├── images/                                 # Visualizations
├── src/                                    # Python scripts (optional)
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

Key Features

Custom Transformer Encoder implementation from scratch
Multi-head self-attention for temporal pattern recognition
Layer normalization and residual connections
Early stopping to prevent overfitting
Complete preprocessing pipeline


Future Work

Experiment with deeper architectures
Test on multivariate time series
Implement hybrid CNN-Transformer models
Applications: predictive maintenance, anomaly detection, ECG classification


References

Bagnall, A., et al. (2018). The UEA multivariate time series classification archive. ArXiv
Vaswani, A., et al. (2017). Attention is All You Need. ArXiv
Chollet, F., et al. (2015). Keras Documentation


Author
Samuel Bulnes
Submission Date: October 3, 2025
GitHub: @Samuel-Bulnes
License
MIT License - see LICENSE file for details.

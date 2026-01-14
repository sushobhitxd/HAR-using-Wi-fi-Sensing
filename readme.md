# **Human Activity Recognition (HAR) using Wi-Fi Sensing**

A device-free Human Activity Recognition system that utilizes Wi-Fi Channel State Information (CSI) signals to detect and classify human movements. 
By analyzing signal perturbations caused by motion, this project provides a privacy-preserving, non-invasive alternative to camera-based monitoring systems.


**🚀 Key Features**

    Hybrid Deep Learning Architecture: Combines 1D CNN for spatial feature extraction (subcarrier correlations) and LSTM for temporal sequence learning.

    Advanced Signal Processing: * Butterworth Filter: Removes high-frequency environmental noise.

        Outlier Suppression: Percentile-based clipping to handle burst interference.

        PCA Reduction: Compresses 90 CSI streams into principal components for efficient real-time processing.

    Real-Time Dashboard: An interactive Streamlit-based web UI for live signal visualization and activity inference.

    High Performance: Achieves ~90% accuracy on the UT-HAR benchmark dataset.


**🛠️ Tech Stack**

    Language: Python 3.8+

    Deep Learning: TensorFlow/Keras

    Signal Processing: SciPy, Scikit-learn (PCA)

    Interface: Streamlit, Matplotlib


**📊 Dataset & Classification**

The system is trained on the UT-HAR (University of Toronto) dataset, which includes time-series CSI amplitude and phase data collected via Intel 5300 NIC.

Supported Activity Classes:

    🏃 Run | 🚶 Walk | 🛌 Lie Down | ⚠️ Fall | 🧺 Pick Up | 🪑 Sit Down | 🧍 Stand Up

**💻 Usage**

1. *Installation*

`git clone https://github.com/sushobhitxd/HAR-using-Wi-fi-Sensing.git `
`cd HAR-using-Wi-fi-Sensing `
`pip install -r requirements.txt`

2. *Launching the Dashboard*

> streamlit run app.py

Access the interface at http://localhost:8501.

3. *Inference Modes*

    Simulation Mode: Select a pre-recorded sample (e.g., Test_Walk_01.csv) from the sidebar to visualize how the model processes signals and predicts activities in real-time.

    Live Mode: Connect to an active Intel 5300 NIC stream. The system will read from the designated buffer file to classify live movements in the room.

**📂 Project Structure**

    app.py: Streamlit application and UI logic.

    train.py: Training script for the CNN-LSTM model.

    preprocessing.py: Core logic for Butterworth filtering and PCA.

    models/: Contains the pre-trained har_cnn_lstm.h5 and saved PCA objects.
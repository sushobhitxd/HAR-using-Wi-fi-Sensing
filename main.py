# Importing libraries
import streamlit as st
import numpy as np
import torch
from model import CNN_LSTM_HAR, HARCSIDataset, train_epoch, eval_epoch  
import pandas as pd
import matplotlib.pyplot as plt

#Loading dataset and models
df = pd.read_csv("combined_dataset.csv")

model = CNN_LSTM_HAR(input_size=40, num_classes=5)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

st.set_page_config(page_title= "HUMAN ACTIVITY RECOGNITION USING WIFI-SENSING", layout='wide', initial_sidebar_state="expanded")

# Sidebar navigation
st.sidebar.title(" Menu ")
menu=st.sidebar.radio("Go to:",
    ["💡Home", "🔍Exploratory Data Analysis", '🗃️Model', '🧠Training setup'])

# 💡Home
if menu== "💡Home":
    st.title("DSA project on HAR USING WIFI SENSING")

    st.header("Project Overview")

    st.subheader("CASE STUDY: HUMAN ACTIVITY RECOGNITION USING WIFI-SENSING")
    st.markdown(
        """
        Human Activity Recognition (HAR) using WiFi-sensing is an innovative approach that leverages the variations in WiFi signals caused by human movements to identify and classify different activities. This method utilizes Channel State Information (CSI) from WiFi signals, which provides detailed insights into how the signal propagates through the environment and interacts with moving objects, including humans.
        """
    )

    st.header("Objective")
    st.markdown(
        """
            The primary objective of this project is to develop a robust machine learning model capable of accurately recognizing and classifying human activities based on WiFi CSI data. By analyzing the patterns and variations in the WiFi signals, the model aims to identify specific activities such as walking, sitting, standing, and other common movements.
        """
    )

    st.header("Dataset Overview")
    st.markdown(
        """
        The dataset used in this project consists of WiFi CSI data collected from various environments and scenarios. Each data point represents the CSI values corresponding to different human activities, along with their respective labels. The dataset is structured to facilitate the training and evaluation of machine learning models for activity recognition.

        **Dataset Source:** Dataset for Human Activity Recognition using Wi-Fi Channel State Information (CSI) data (https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892/1?file=27485900)
             
        """
    )

# 🔍Exploratory Data Analysis
elif menu== "🔍Exploratory Data Analysis":
    st.title("DATASET EXPLORATION")

    st.divider()

    st.header("Raw Datatset")

    st.markdown("""
    The dataset is organised as follows:
    * There are 3 folders (room_1/2/3), which indicate data collected from 3 different rooms
    * In each of that rooms there is a different number of folders indicating different data capturing session
    * In each of that folder, there is data.csv files, which stores CSI data for each packet. Also, there are label.csv and label_boxes.csv which contain labels for activities and person bounding box respectively.
    """)

    DATASET_FILE = 'wifi_csi_har_dataset/room_1/1/data.csv' 
    CLEANED_DATASET_FILE = 'wifi_csi_har_dataset/room_1/1/cleaned_data.csv' 
    #  Load the data 
    @st.cache_data
    def load_data(file_path):
        data = pd.read_csv(file_path)
        return data

    @st.cache_data
    def load_cleaned_data(file_path):
        data = pd.read_csv(file_path)
        return data

    try:
        df = load_data(DATASET_FILE)
        
        st.subheader("Dataset Preview: Top 5 Rows")
        st.write(f"The full dataset contains {len(df)} rows and {len(df.columns)} columns. This is a preview of the first five entries:")
        
        st.dataframe(df.head(5))

    except FileNotFoundError:
        st.error(f"Error: The file '{DATASET_FILE}' was not found. Please ensure it is in the same directory as your Streamlit script.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")

    st.divider()    
    st.header(" Analysis before Cleaning")

    st.header("Amplitude Time Series")

    st.image("Images/uncleaned_1.png",
            caption="Figure 1: CSI Amplitude Time series")


    st.header("Heatmap of the Dataset")

    st.image("Images/uncleaned_2.png",
            caption="Figure 2: CSI Heatmap")

    st.header("Activity Label Distribution")

    st.image("Images/uncleaned_3.png",
            caption="Figure 3: Activity Distribution")

    st.divider()

    st.header(" Analysis after Cleaning")  

    st.header("PCA feature Correlation")
    st.image("Images/cleaned_1.png",
            caption="Figure 4: PCA feature Correlation after Cleaning")
    st.header("Boxplot for the PCA features")
    st.image("Images/cleaned_2.png",
            caption="Figure 4: Boxplot for the PCA feature distribution after Cleaning")



# 🗃️Model
elif menu == "🗃️Model":
    st.title("The Activity Layer")
    st.subheader("Machine Learning Models")
    st.divider()

    st.header("Model choices and Rationale")
    st.markdown("""
    Human Activity Recognition (HAR) with WiFi Channel State Information (CSI) presents unique challenges due to the complex, high-dimensional, and sequential nature of the data. To address these challenges, various deep learning architectures can be employed, each offering distinct advantages:
                """)
                
    st.subheader("*Convolutional Neural Network(CNN)*")
    st.markdown("""
                CNNs are highly effective for extracting local spatial features from high-dimensional data. When applied to CSI matrices, 1D convolutions treat each time step as an observation across multiple WiFi subcarriers, enabling the network to learn subtle correlations and distinctive signal patterns linked to human movements. CNNs excel at identifying spatial dependencies, making them ideal for capturing static or local structure within the signal.
    """)

    st.divider()

    st.subheader("RNN ~ LSTM (lONG-SHORT TERM MEMORY)")

    st.markdown("""
                LSTMs, a type of recurrent neural network, are designed to model temporal dynamics in sequential data. By maintaining hidden states and memory cells, LSTMs can capture both short-term changes and long-term dependencies within the CSI signal as human activities unfold. This allows recognition of activities that develop over time or involve transitions, such as walking or sitting down.
""")

    st.divider()


    st.header("CNN + LSTM Hybrid ")

    st.markdown("""
    To fully leverage both the spatial and temporal information inherent in CSI data, our project adopts a hybrid CNN + LSTM architecture. In this design:

    - **CNN layers first extract spatial patterns and correlations between adjacent subcarriers at each time step, creating compact feature maps that highlight key movement cues.**
    - **LSTM layers then process the sequence of these spatial features, learning how movements evolve and change over time.**
    - **A final fully connected classification layer maps the learned spatiotemporal patterns to discrete activity classes.**

    ---

    **Advantages of the Hybrid Approach:**

    The hybrid CNN-LSTM model provides a robust solution that combines the representational strengths of both architectures. By integrating spatial feature extraction with temporal sequence modeling, it achieves:

    - Superior recognition accuracy over standalone CNN or LSTM models
    - Enhanced ability to differentiate between complex or similar activities
    - Improved generalization to real-world HAR scenarios

    This dual-stage learning strategy allows for more accurate, reliable, and privacy-preserving inference of human activity from WiFi CSI, supporting applications in smart environments, healthcare, and ambient assisted living.""")

    st.subheader("Model Hyperparameters:")
    st.markdown("""
    - **CNN Conv1D filters:** [32, 64]
    - **Kernel size:** 3
    - **Dropout:** 0.1
    - **LSTM units:** 64
    - **LSTM layers:** 1
    - **Fully connected layer:** 100 neurons
    - **Optimizer:** Adam (lr = 0.001)
    - **Batch size:** 8
    - **Early stopping patience:** 15
    """)

    test_accuracy = 0.75
    macro_f1 = 0.76
    weighted_f1 = 0.76

    st.markdown(
        f"""
        <div style="background-color:#222;padding:16px;border-radius:12px;">
        <span style="color:#aef3c2;font-size:20px;">Accuracy: <b>{test_accuracy:.2f}</b></span><br>
        <span style="color:#7fcdfc;font-size:20px;">Macro F1 Score: <b>{macro_f1:.2f}</b></span><br>
        <span style="color:#f3cea7;font-size:20px;">Weighted F1 Score: <b>{weighted_f1:.2f}</b></span>
        </div>
        """, unsafe_allow_html=True
    )

    st.divider()


# 🧠Training setup
elif menu == "🧠Training setup":
    st.title("Training Setup")
    st.markdown("""
    Efficient training is the backbone of any deep learning pipeline, directly affecting the performance and reliability of Human Activity Recognition (HAR) models. The training setup in this project is designed to optimize learning, promote generalization, and prevent overfitting. Here’s how each component works:
    """)

    st.header("Loss Function")
    st.subheader("""
    *Purpose*
                """)
    st.markdown("""
    The loss function measures the mismatch between the model’s predictions and the actual activity labels. For classification tasks, we use CrossEntropyLoss, which evaluates how well the predicted class probabilities match the true classes.""")

    st.subheader("""
    *Importance*
                """)
    st.markdown("""
    Minimizing the loss during training guides the neural network to improve its predictions, continually adjusting the weights for better accuracy.""")

    st.subheader("""
    *In This Pipeline*
                """)
    st.markdown("""
    At each step, the model compares its outputs for different activities to the true encoded labels and calculates the loss. Training aims to reduce this value as much as possible.
    """)

    st.divider()


    st.header("Optimizer")
    st.subheader("""
    *Purpose*
                """)
    st.markdown("""
    The optimizer updates the model’s internal weights based on computed gradients from the loss function, driving the learning process.""")

    st.subheader("""
    *Common Choice*
                """)
    st.markdown("""
    Adam optimizer is selected for its speed and stability in complex deep learning models. Adam dynamically adjusts learning rates for each parameter, promoting faster convergence.""")

    st.subheader("""
    *Configuration*
                """)
    st.markdown("""
    A learning rate of 0.001 is used, striking a balance between progress and stability. This controls how quickly model weights are updated.""")


    st.divider()


    st.header("Training Loop")
    st.subheader("""
    *Purpose*
                """)
    st.markdown("""
    The training loop orchestrates the process—feeding batches of data into the model, computing loss, performing backpropagation, and updating weights over many epochs.""")

    st.subheader("""
    *Batch Training*
                """)
    st.markdown("""
    Data is split into small batches (batch size: 8) for efficiency. Batch training smooths gradient updates and accelerates convergence.""")
    st.divider()

    st.header("Early Stopping")
    st.subheader("""
    *Purpose*
                """)
    st.markdown("""
    Prevents over-training the model and helps avoid memorizing training data ("overfitting").""")

    st.subheader("""
    *How It Works*
                """)
    st.markdown("""
    Validation accuracy is continuously monitored. If performance does not improve for a set number of epochs (patience: 15), training halts, and the best model state is preserved.""")

    st.subheader("""
    *Benefit*
                """)
    st.markdown("""
    Early stopping selects models that generalize well to new, unseen activity data, ensuring robust real-world results.""")

    st.divider()
    st.header("Why are these Steps Important ?")
    st.subheader("""
    *Generalization*
                """)

    st.markdown("""
    Validating on separate data and applying early stopping prevents memorization and encourages learning meaningful, transferable patterns.
    - **Efficiency & Reproducibility:**  
    Consistent training with defined loss functions, optimizers, and batch processing ensures your results are scientific, reproducible, and robust.
    - **Seamless Integration:**  
    These training elements connect model building to evaluation, bridging development and deployment workflows.
                
    Loss function, optimizer, batch training, and early stopping collectively drive your HAR model to learn meaningful activity feat            """) 
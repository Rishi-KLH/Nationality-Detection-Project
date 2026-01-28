# Nationality-Detection-Project
venv/
__pycache__/
*.pyc
.DS_Store
FairFace/train/
FairFace/val/

#Project Title
Smart Demographic Attribute Prediction using Deep Learning: Nationality Group, Emotion, Age and Dress Color Recognition from Facial Images

#Abstract
This project focuses on building an AI-based system that can identify a person’s nationality group from an uploaded image and also predict their emotion. The model is trained using the FairFace dataset and a deep learning approach (ResNet-18 transfer learning) to classify facial features. After nationality prediction, the system follows specific rules to display extra details: if the person is Indian, it also predicts age and dress colour along with emotion; if the person is from the United States, it predicts only age and emotion; if the person is African, it predicts emotion and dress colour; and for other nationalities, it shows only nationality and emotion. Dress colour is detected using dominant colour extraction, and prediction confidence is used to avoid incorrect results. The complete project is implemented with a Streamlit GUI where users can upload an image, preview it, and view the final output clearly.

#Problem Statement
The aim of this project is to develop an AI model that predicts a person’s nationality group and emotion from an uploaded image. Based on the nationality, the system should conditionally predict additional attributes like age and/or dress colour. A GUI must be created to allow users to upload an image, preview it, and display the prediction results clearly.

#Dataset
For this project, the FairFace dataset is used. FairFace is a large-scale face image dataset created to support research in demographic analysis. It contains face images labeled with attributes such as race/ethnicity, age group, and gender.

In this project, the race labels from FairFace are used to classify nationality groups such as Indian, United States, African, and Other. The dataset provides separate training and validation folders along with CSV label files, which helps in building and evaluating the deep learning model effectively.

#output commands
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python train_nationality.py


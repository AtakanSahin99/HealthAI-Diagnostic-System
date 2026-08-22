# HealthAI Diagnostic System

## Overview
HealthAI is a multi-modal healthcare decision support system designed to assist medical professionals by combining structured clinical patient records and unstructured radiological image data. The platform provides predictive risk scores for stroke occurrences and automated classification of Brain MRI scans.

## Features
- Structured clinical data analysis using tabular machine learning algorithms.
- Stroke risk prediction based on patient health metrics such as age, glucose level, BMI, and hypertension history.
- Synthetic Minority Over-sampling Technique (SMOTE) implementation to address dataset class imbalance.
- Deep learning image classification model utilizing Convolutional Neural Networks (CNN) for Brain MRI scans.
- Interactive Streamlit dashboard providing immediate visual analytics, confusion matrices, and ROC curves.
- Privacy-conscious stateless operational architecture without persistent raw patient data storage.

## Technologies
- Python 3.1x
- Streamlit
- TensorFlow / Keras
- Scikit-Learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Imbalanced-Learn (SMOTE)

## Project Structure
`
HealthAI-Diagnostic-System/
├── src/
│   ├── main_app.py
│   ├── data_pipeline/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── model_factory/
│   │   ├── trainer.py
│   │   └── image_trainer.py
│   └── visualization/
│       └── charts.py
├── data/
│   ├── stroke.csv
│   └── brain_mri/
├── assets/
│   └── images/
├── docs/
│   └── report/
│       ├── HEALTHAI_Report.docx
│       └── REPORT.md
├── requirements.txt
├── .gitignore
└── README.md
`

## Installation
1. Clone the repository to your local environment.
2. Ensure Python 3.10+ is installed.
3. Install required dependencies:
`ash
pip install -r requirements.txt
`

## Usage
Launch the Streamlit web application:
`ash
streamlit run src/main_app.py
`
Navigate to the provided local URL (typically http://localhost:8501) to interact with the clinical stroke risk calculator and Brain MRI classifier tabs.

## Screenshots
Screen captures detailing UI components, model evaluation charts, and diagnostic outputs are located in the ssets/images/ directory.

## Documentation
The comprehensive Turkish technical report is available in Markdown format at docs/report/REPORT.md and in DOCX format at docs/report/HEALTHAI_Report.docx.

## Future Improvements
- Integration of DICOM image format support for radiological standard compatibility.
- Deployment to cloud infrastructures with FHIR/HL7 hospital information system interfaces.
- Inclusion of real-time wearable device telemetry for continuous patient risk monitoring.

## License
Educational and research project published for academic purposes.

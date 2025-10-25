# Computer-aided-lung-cancer-diagnosis-using-Explainable-AI
This project is a computer-aided diagnosis system for lung cancer that uses deep learning and explainable AI (XAI) to classify lung tissue images. The system is deployed as a Flask web server and can be accessed through a mobile application, providing a practical tool for early detection and interpretation of lung cancer. 

---


##  Highlights

-  **Model**: VGG16 CNN fine-tuned on LC25000 dataset  
-  **Explainability**: Grad-CAM heatmaps for visual interpretation  
-  **Deployment**: Flask-based cloud server + mobile app integration  
-  **Accuracy**: 97.6% training, 95.9% validation  
-  **Dataset**: 15,000 histopathological images across 3 lung cancer classes  

---

##  Motivation

Lung cancer is one of the deadliest diseases globally, often diagnosed too late. This project aims to assist clinicians with early detection using AI, while ensuring transparency through visual explanations. Grad-CAM helps highlight regions of interest, making the model's decisions interpretable and trustworthy.

---

##  Tech Stack

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| TensorFlow/Keras | Deep learning model training     |
| Grad-CAM         | Explainable AI visualizations    |
| Flask            | Web deployment                   |
| NumPy & Matplotlib | Data processing & visualization |
| scikit-learn     | Evaluation metrics               |

---

##  Dataset

- **Source**: LC25000  
- **Classes**: Normal, Adenocarcinoma, Squamous Cell Carcinoma  
- **Split**: 12,000 training images, 3,000 testing images  
- **Resolution**: Resized to (122, 122, 3) for model input  


---

##  Model Architecture

The model uses VGG16 with transfer learning. Here's a simplified architecture diagram:


---

##  Grad-CAM Visualization

Grad-CAM highlights the regions of the image that influenced the model’s prediction. This helps clinicians validate the AI's decision.

![Grad-CAM Heatmaps](src/images/gradcam_result_n.png)
![Grad-CAM Heatmaps](src/images/gradcam_result_scc.png)

---

##  Training & Evaluation Results

###  Accuracy and Loss Curves

- **Training Accuracy**: 97.6%  
- **Validation Accuracy**: 95.9%  
- **Training Loss**: 0.2227  
- **Validation Loss**: 0.2325  

![Accuracy and Loss Curves](src/images/cls_report.png)

###  Confusion Matrix & Classification Report

- High precision across all three classes  
- Diagonal values: Normal (998), Adenocarcinoma (955), Squamous Cell (941)

![Confusion Matrix and Report](src/images/cf_matrix.png)

---

##  Mobile App Interface

Users can upload lung tissue images via a mobile interface. The app communicates with the Flask server to receive classification results and Grad-CAM visualizations instantly.

<p align="center">
  <img src="src/images/lc_ACA.jpg" alt="Adenocarcinoma Output" width="45%" />
  <img src="src/images/lc_Normal.jpg" alt="Normal Tissue Output" width="45%" />
</p>

---


##  References

1. [Non-small cell lung cancer diagnosis with explainable deep learning](https://www.sciencedirect.com/science/article/pii/S0169260722004898)  
2. [Explainable AI in medical imaging: Saliency-based approaches](https://www.sciencedirect.com/science/article/pii/S0720048X23001018)

---

_“AI should not just be accurate—it should be understandable.”_


# 🔋 Battery Health Prediction

As lithium-ion batteries are used over time, their **capacity gradually degrades** due to aging and cycling. When the battery's capacity drops below a critical threshold, it reaches **End-of-Life (EOL)** and must be replaced for safety and reliability. 

This project focuses on **Remaining Useful Life (RUL) prediction** for Li-ion batteries.

---

## 📦 Features

- 🔁 Predict RUL from capacity degradation curves over charging cycles
- 📊 Trained and evaluated on the **NASA Battery Aging Dataset**
- 🤖 Implements a **LSTM-AutoEncoder** for unsupervised feature extraction and dimensionality reduction
- 🧠 Includes reproduction of deep learning models from recent research

---

## 📂 Dataset

**Li-ion Battery Aging Dataset**:

🔗 [NASA Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

This dataset includes time series measurements (voltage, current, temperature, and capacity) from multiple batteries under different operating conditions.

---

## 📚 Referenced Papers

1. **Khaleghi, S., Hosen, M. S., Van Mierlo, J., Berecibar, M.**  
   *Towards machine-learning driven prognostics and health management of Li-ion batteries: A comprehensive review*  
   **Renewable and Sustainable Energy Reviews**, 2023  
   DOI: [10.1016/j.rser.2023.114224](https://doi.org/10.1016/j.rser.2023.114224)  
   → A thorough survey of machine learning techniques used for battery State of Health (SoH) estimation and RUL prediction, highlighting current challenges and future directions.

2. **Li, P., Zhang, Z., Xiong, Q., Ding, B., Hou, J., Luo, D., Rong, Y., Li, S.**  
   *State-of-health estimation and remaining useful life prediction for the lithium-ion battery based on a variant long short term memory neural network*  
   **Journal of Power Sources**, Volume 480, 2020, 228069  
   DOI: [10.1016/j.jpowsour.2020.228069](https://doi.org/10.1016/j.jpowsour.2020.228069)  
   → Proposes the AST-LSTM model, which improves traditional LSTM architectures by incorporating attention mechanisms for more accurate SoH and RUL prediction.

---

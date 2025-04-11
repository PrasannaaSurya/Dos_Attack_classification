# Dos_Attack_classification
**Dos_Attack_classification** is a deep learning-based web application that detects and classifies Denial of Service (DoS) attacks in real-time network traffic. The app is built using **TensorFlow** and **Streamlit**, combining machine learning algorithms and a user-friendly interface to enhance cybersecurity through intelligent automation.

## 🔐 Features

- Detects and classifies:
  - 🛡️ Benign Traffic
  - 🚫 DoS Hulk
  - 🐢 DoS Slowloris
- Streamlit-based interface for real-time classification
- Login and registration system (`auth.py`)
- Real-time attack statistics stored in `stats.json`
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Model comparison: Random Forest, SVM, Decision Tree, Naive Bayes

## 🧪 Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- StandardScaler, LabelEncoder
- PyCharm (Development IDE)

## 📊 Input Features

- `Destination_Port`
- `Total_Fwd_Packets`
- `Total_Backward_Packets`
- `Total_Length_of_Fwd_Packets`
- `Flow_Bytes_Sec`
- `Flow_Packets_Sec`
- `Fwd_IAT_Total`
- `Packet_Length_Std`
- `Packet_Length_Variance`

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Dos_Attack_classification.git
   cd Dos_Attack_classification
   ```

2. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run check.py
   ```

## 📌 Future Enhancements

- Add more attack types (e.g., DDoS, Port Scan)
- Alert-based mitigation strategies
- Visual dashboard for analytics
- Cloud deployment (Streamlit Cloud, AWS, etc.)

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request for improvements or bug fixes.

## 📜 License

[MIT](LICENSE)

**AI-Powered Threat Detection System for Network Anomalies**

**Introduction**

This project focuses on developing a threat detection system using machine learning, specifically by identifying anomalies in network traffic data. The goal is to detect potentially malicious activities by recognizing unusual patterns that deviate from normal network behavior. This is achieved through an unsupervised anomaly detection technique using the Isolation Forest algorithm.
Problem Statement
The increasing sophistication and frequency of cyberattacks necessitate robust and automated threat detection mechanisms. Traditional signature-based methods often struggle to identify novel or zero-day attacks. This project addresses the need for a system capable of detecting anomalies in network traffic, which can be indicative of various security threats, such as intrusions, malware infections, or denial-of-service attacks.

**Objectives**

•	Develop a threat detection system using the Isolation Forest machine learning algorithm.
•	Evaluate the effectiveness of Isolation Forest in identifying anomalies in network traffic data.
•	Gain insights into network traffic patterns for threat detection purposes.

**Dataset**

The project utilizes network traffic data stored in CSV format. The training data is contained in a file named network_traffic.csv, and the testing data is in Anomaly_test.csv. The datasets include features like Time, Length, Source, Destination, Protocol, and Info. These datasets are assumed to be unlabeled, as the approach is unsupervised anomaly detection.

**Methodology**

This project employs the Isolation Forest algorithm for anomaly detection. Isolation Forest works by isolating anomalies rather than profiling normal data points. It randomly selects a feature and then randomly selects a split value between the maximum and minimum values of that feature. This partitioning of instances is represented as a tree structure. Anomalies are expected to have shorter paths in the tree (i.e., they are isolated closer to the root) because they have attribute-values that are very different from normal instances.

The following steps are performed:
1.	Data Loading and Preprocessing:
o	The CSV data is loaded using the pandas library (pd.read_csv).
o	Duplicate rows are removed using drop_duplicates().
o	Rows with missing values are removed using dropna().
o	The categorical Protocol column is converted into numerical data using one-hot encoding with pd.get_dummies(columns=['Protocol'], drop_first=True). The drop_first=True argument avoids multicollinearity.
o	The numerical Time and Length features are scaled using MinMaxScaler to normalize their ranges between 0 and 1. This prevents features with larger values from dominating the model.
o	The Source and Destination columns are hashed using SHA-256 to convert IP addresses (or other string representations) into numerical values. This also anonymizes the data. The modulo operator (%) is used to keep the hashed values within a manageable range (0-107).
2.	Training-Testing Split (Training Data):
o	The preprocessed training data is split into training and testing sets using train_test_split with a test size of 20% (test_size=0.2) and a fixed random state (random_state=42) for reproducibility.
3.	Model Training:
o	An IsolationForest model is instantiated with 100 estimators (n_estimators=100), a contamination parameter of 0.05 (assuming 5% of the data is anomalous), and a fixed random state (random_state=42).
o	The model is trained on the training data using model.fit(X_train).
4.	Testing and Evaluation (Testing Data):
o	The test data is loaded from Anomaly_test.csv.
o	The same preprocessing steps (duplicate removal, missing value handling, one-hot encoding, scaling, and hashing) are applied to the test data to ensure consistency with the training data.
o	Crucially, the test data is reindexed to align its columns with the training data columns using test_data.reindex(columns=X_train.columns, fill_value=0). This is essential because the one-hot encoding might create different columns in the test set if different protocol values are present. The fill_value=0 handles cases where a protocol present in the training data is absent in the test data.
o	The trained MinMaxScaler (scaler) is used to transform the test data’s Time and Length features.
o	The trained model is used to predict anomalies in the test data using model.predict(test_data). The predictions are either -1 (anomaly) or 1 (normal).
5.	Results Visualization:
o	The distribution of predictions (number of anomalies and normal instances) is visualized using bar charts using matplotlib.pyplot.

 
The bar chart visualizes the results of your anomaly detection model on the test dataset (Anomaly_test.csv):
<img width="277" alt="image" src="https://github.com/user-attachments/assets/7fe9fde3-ee08-4b9d-a352-4c4da6d2ae73" />



**Network Anomaly Detection Dashboard:**

•	Purpose: This section should explain the role of the Streamlit dashboard within your project. Briefly mention that it provides a user-friendly interface for real-time anomaly detection on new network traffic data.

•	Functionality: Describe the key features of the dashboard: 

•	File Upload: Users can upload CSV files containing network traffic data for anomaly prediction.

•	Preprocessing: The uploaded data is preprocessed using the same steps applied during model training (handling missing values, duplicates, one-hot encoding protocols, scaling numerical features, hashing IP addresses, and reindexing columns).

•	Prediction: The preprocessed data is fed into the trained Isolation Forest model to generate anomaly predictions (-1 for anomaly, 1 for normal).

•	Results: The dashboard displays various metrics: 

•	Total records processed.

•	Number of anomalies detected.

•	Number of normal instances identified.

•	Visualization: A pie chart visually represents the distribution of normal and anomalous traffic.

•	Anomaly Details: 
	-The original data is augmented with a new 'Anomaly' column based on the predictions.
  -A table displays the rows classified as anomalies, allowing users to inspect these instances.
  -An option to download the anomalies data as a CSV file is provided.

On uploading a csv file(which was converted from a pcap file):
  
<img width="468" alt="image" src="https://github.com/user-attachments/assets/d1a943e0-52de-421d-885e-7fa572c5c210" />
<img width="468" alt="image" src="https://github.com/user-attachments/assets/24527676-5b07-4267-b996-624cf9a9abb2" />

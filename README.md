# astronaut-wellness-model
🚀 Astronaut Wellness Prediction Model
A comprehensive machine learning project developed during my Data Science Internship at Celebal Technologies, aimed at predicting astronaut health outcomes based on mission-related biomedical inputs. This model can predict psychological risk levels and bone density loss, and also provide personalized recommendations for better space mission planning.

📌 Project Objectives
Predict whether an astronaut is at psychological risk (binary classification).

Predict the percentage of bone density loss (regression).

Provide recommendations based on critical physiological inputs.

Visualize the relationships between space health metrics using a range of visual tools.

🧰 Tech Stack & Libraries Used
Tool/Library	Purpose
Pandas	Data manipulation and DataFrame operations
Seaborn	Advanced and customizable data visualization
Matplotlib	Basic plotting and figure control
Scikit-learn	Preprocessing, training machine learning models

📊 Visualizations
To understand relationships and outliers in the dataset, the following plots were used:

✅ 1. Heatmap
Shows correlation between features.

Helps identify multicollinearity or potential predictors.

python
Copy
Edit
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
✅ 2. Histogram
Used to display the distribution of individual features such as Radiation Exposure or Muscle Mass Loss.

python
Copy
Edit
data['Radiation_Exposure (mSv)'].hist(bins=20)
✅ 3. Bar Plot
Visualizes categorical comparisons such as Mission Count vs Gender or Vision Change.

python
Copy
Edit
sns.barplot(x='Gender', y='Muscle_Mass_Loss (%)', data=data)
✅ 4. Line Plot
Tracks trends over mission duration.

python
Copy
Edit
sns.lineplot(x='Mission_Duration (days)', y='Bone_Density_Loss (%)', data=data)
✅ 5. Scatter Plot
Helps visualize interaction between two continuous variables.

python
Copy
Edit
sns.scatterplot(x='Radiation_Exposure (mSv)', y='Immune_Response_Suppression (%)', hue='Gender', data=data)
✅ 6. Box Plot
Identifies outliers and variance in features.

python
Copy
Edit
sns.boxplot(x='Gender', y='Bone_Density_Loss (%)', data=data)
🤖 Machine Learning Models
✅ 1. Decision Tree Classifier
Used to classify astronauts as either High Risk or Low Risk for psychological impact.

Interpretability makes it ideal for health-related domains.

python
Copy
Edit
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
✅ 2. Random Forest Classifier
An ensemble method that increases accuracy and reduces overfitting.

Used to improve classification performance over the basic Decision Tree.

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
✅ 3. Linear Regression
Used to predict the continuous output: percentage of bone density loss.

Assesses how mission duration, radiation, and muscle mass loss affect bone density.

python
Copy
Edit
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
🧪 Evaluation Metrics
Model	Metric(s) Used
Classifiers	Accuracy, Precision, Recall, F1-score
Regression	R² Score, Mean Absolute Error (MAE)

Example:

python
Copy
Edit
from sklearn.metrics import accuracy_score, r2_score
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("R² Score:", r2_score(y_test, y_pred_reg))
🧠 Sample Input Features
Feature Name	Description
Age_at_Mission_Start	Age of astronaut at mission start
Gender	Encoded gender
Country	Encoded nationality
Mission_Duration (days)	Duration of space mission
Number_of_Missions	Past space missions participated
Muscle_Mass_Loss (%)	Percentage loss of muscle mass
Radiation_Exposure (mSv)	Exposure to space radiation
Cardiovascular_Risk_Score	Score between 1-10
Immune_Response_Suppression (%)	Immune system weakening estimate
Vision_Change	Change in vision (e.g. Mild, Severe)
Countermeasures_Used	Exercises or supplements used

📌 Key Highlights
🔬 Predicts space-induced health deterioration with actionable output.

🛠️ Supports personalized recommendations based on input thresholds.

📊 Strong focus on interpretability and visualization.

📚 A valuable resource for applying data science in aerospace health domains.

📁 Folder Structure (Suggested)
bash
Copy
Edit
astronaut-wellness-model/
│
├── data/                    # Raw or cleaned datasets
├── visuals/                 # Heatmaps, plots, graphs
├── models/                  # Trained model files (.pkl)
├── notebook/                # Jupyter Notebooks
├── src/                     # Python scripts (modularized)
│
├── README.md                # You're reading it!
├── requirements.txt         # Dependencies
├── LICENSE
🔮 Future Enhancements
Integrate with wearable sensor data in real-time.

Convert to Streamlit Web App for live predictions.

Explore clustering (unsupervised) for health pattern discovery.

🙌 Acknowledgements
Gratitude to Celebal Technologies for the opportunity to work on cutting-edge data science projects in the field of astronaut health and analytics. Thanks to mentors and teammates for their guidance!

📬 Let's Connect
Feel free to reach out on LinkedIn for collaborations or feedback.
linkedid link: www.linkedin.com/in/shalinikumari7206


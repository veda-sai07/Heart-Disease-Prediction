# ==========================================================
# SMART AI-BASED CARDIOVASCULAR DISEASE PREDICTION SYSTEM
# Dataset : heart.csv
# Target  : cardio
# ==========================================================

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

# ==========================================================
# CREATE SCREENSHOTS FOLDER
# ==========================================================

os.makedirs("screenshots", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("SMART HEART DISEASE PREDICTION SYSTEM")
print("=" * 60)

df = pd.read_csv("heart.csv", sep=";")

print("\nDataset Shape :", df.shape)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

df["age_years"] = (df["age"] / 365).astype(int)

df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

# Remove unnecessary columns
df.drop(["id", "age"], axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("Dataset Shape After Cleaning :", df.shape)

# ==========================================================
# BASIC DATA ANALYSIS
# ==========================================================

print("\nDataset Summary")
print(df.describe())

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("screenshots/correlation_matrix.png")
plt.show()

# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6, 4))
sns.countplot(x="cardio", data=df)
plt.title("Heart Disease Distribution")
plt.savefig("screenshots/target_distribution.png")
plt.show()

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("cardio", axis=1)
y = df["cardio"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "SVM": SVC(probability=True)
}

results = {}
times = {}

print("\nMODEL TRAINING STARTED...\n")

# ==========================================================
# MODEL TRAINING
# ==========================================================

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy
    times[name] = end - start

    print(f"{name} ---> {accuracy*100:.2f}%")

# ==========================================================
# MODEL LEADERBOARD
# ==========================================================

leaderboard = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy (%)": [round(v * 100, 2) for v in results.values()]
})

leaderboard = leaderboard.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\n🏆 MODEL LEADERBOARD")
print(leaderboard)

# ==========================================================
# ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8, 5))
sns.barplot(
    x="Model",
    y="Accuracy (%)",
    data=leaderboard
)

plt.xticks(rotation=20)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("screenshots/model_accuracy.png")
plt.show()

# ==========================================================
# BEST & FASTEST MODEL
# ==========================================================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

fastest_model = min(times, key=times.get)

print("\nMost Accurate Model :", best_model_name)
print("Fastest Model       :", fastest_model)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("screenshots/confusion_matrix.png")
plt.show()

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test, pred))

# ==========================================================
# ROC CURVE
# ==========================================================

probabilities = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("screenshots/roc_curve.png")
plt.show()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n🔥 TOP 5 RISK FACTORS")
print(importance.head())

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("screenshots/feature_importance.png")
plt.show()

# ==========================================================
# SAMPLE PREDICTION
# ==========================================================

sample = X.iloc[0].values.reshape(1, -1)

sample = scaler.transform(sample)

probability = best_model.predict_proba(sample)[0][1]

# ==========================================================
# HEALTH SCORE
# ==========================================================

health_score = int((1 - probability) * 100)

if probability < 0.30:
    risk = "LOW"
elif probability < 0.70:
    risk = "MEDIUM"
else:
    risk = "HIGH"

# ==========================================================
# LIFESTYLE SCORE
# ==========================================================

risk_score = 0

if probability > 0.7:
    risk_score += 40

if health_score < 50:
    risk_score += 30

risk_score = min(risk_score, 100)

fitness_score = 100 - risk_score

if fitness_score >= 80:
    rating = "Excellent"
elif fitness_score >= 60:
    rating = "Good"
elif fitness_score >= 40:
    rating = "Average"
else:
    rating = "Poor"

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(best_model, "heart_model.pkl")

# ==========================================================
# FINAL DASHBOARD
# ==========================================================

print("\n")
print("=" * 60)
print("SMART AI HEART HEALTH DASHBOARD")
print("=" * 60)

print("Disease Risk      :", round(probability * 100, 2), "%")
print("Risk Level        :", risk)
print("Health Score      :", health_score, "/100")
print("Lifestyle Score   :", risk_score, "/100")
print("Fitness Rating    :", rating)
print("Best Model        :", best_model_name)
print("Fastest Model     :", fastest_model)

print("\nTop Risk Factors")

for _, row in importance.head(5).iterrows():
    print("•", row["Feature"])

print("\nRecommendations")

if risk == "HIGH":
    print("✓ Consult Cardiologist")
    print("✓ Reduce Cholesterol")
    print("✓ Daily Exercise")
    print("✓ Monitor Blood Pressure")

elif risk == "MEDIUM":
    print("✓ Improve Diet")
    print("✓ Regular Exercise")
    print("✓ Periodic Health Checkups")

else:
    print("✓ Maintain Healthy Lifestyle")
    print("✓ Continue Physical Activity")

print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)



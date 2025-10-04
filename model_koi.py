import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Read the dataset & Create df_selected
df_koi = pd.read_csv("datasets\cumulative_2025.09.21_10.28.10.csv", comment="#")

selected_features = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_depth', 'koi_duration', 'koi_impact', 'koi_period', 'koi_model_snr',
    'koi_prad', 'koi_teq', 'koi_steff', 'koi_srad', 'koi_slogg', 'koi_disposition'
]
df_selected = df_koi[selected_features]


# Step 2: Fill NaNs with mean using SimpleImputer
numerical_features = [
    'koi_depth', 'koi_impact', 'koi_model_snr', 'koi_prad', 'koi_teq',
    'koi_steff', 'koi_srad', 'koi_slogg'
]
imputer = SimpleImputer(strategy='mean')
df_selected[numerical_features] = pd.DataFrame(
    imputer.fit_transform(df_selected[numerical_features]),
    columns=numerical_features,
    index=df_selected.index
)

# Save the fitted imputer
joblib.dump(imputer, 'imputer.pkl')
print("Imputer saved as 'imputer.pkl'")


# Step 3: Drop remaining NaNs
df_selected = df_selected.dropna()
print(f"Rows after dropping NaNs: {len(df_selected)}")

# Step 4: Map target (koi_disposition)
df_selected['target'] = df_selected['koi_disposition'].map({
    'CONFIRMED': 0, 'CANDIDATE': 1, 'FALSE POSITIVE': 2
})

# Step 5: One-hot encoding (not needed, all predictive features numerical)
print("Feature dtypes:\n", df_selected.dtypes)  # Verify no categoricals

# Step 6: Split x and y
x = df_selected.drop(['koi_disposition', 'target'], axis=1)
y = df_selected['target']

# Check class distribution
print("\nClass Distribution:\n", y.value_counts(normalize=True))

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# Step 8: Train LightGBM
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1
}
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)


# Step 8: Save the model and feature names
joblib.dump(model, 'exoplanet_model.pkl')
print("Model saved as 'exoplanet_model.pkl'")
joblib.dump(x.columns.tolist(), 'feature_names.pkl')
print("Feature names saved as 'feature_names.pkl'")

# Step 9: Evaluate
y_pred = model.predict(X_test)
print("\nTest Set Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, target_names=['Exoplanet', 'Candidate', 'False Positive']))

# Cross-validation
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, x, y, cv=cv, scoring='f1_macro')
print("\nCV F1 Scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())

# Save the model
joblib.dump(model, 'exoplanet_model.pkl')
print("Model saved as 'exoplanet_model.pkl'")

# Also save the feature names for reference
joblib.dump(x.columns.tolist(), 'feature_names.pkl')
print("Feature names saved as 'feature_names.pkl'")



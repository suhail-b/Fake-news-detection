import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load dataset
CSV_FILE_PATH = "IFND.csv"  # Update with correct path
df = pd.read_csv(CSV_FILE_PATH, encoding="ISO-8859-1")

# Define features and target
text_columns = ["Statement", "Web"]  # Text features
categorical_columns = ["Category"]   # Categorical feature
target_column = "Label"  # Target column

# Keep only required features
df = df[text_columns + categorical_columns + [target_column]].dropna()

# Split features and target
X = df[text_columns + categorical_columns]
y = df[target_column]

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Define transformers
text_transformer = TfidfVectorizer(max_features=1000)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Create column transformer
col = ColumnTransformer(transformers=[
    ("tfidf_stmt", text_transformer, "Statement"),
    ("tfidf_web", text_transformer, "Web"),
    ("onehot_cat", categorical_transformer, ["Category"])
])

# Create a pipeline with preprocessing + classifier
pipeline = Pipeline([
    ("col", col),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model and encoders
joblib.dump(pipeline, "model.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Model trained and saved successfully!")



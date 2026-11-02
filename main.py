import pandas as pd
from pathlib import Path

# Dataset location
file_path = Path("data/crx.data")

# Check whether the dataset exists
if not file_path.exists():
    print("Dataset not found!")
    print("Please place crx.data inside the data folder.")
    raise SystemExit(1)

# Load the dataset
df = pd.read_csv(
    file_path,
    header=None,
    na_values="?"
)

# Display the first five records
print("\n--- First Five Records ---")
print(df.head())

# Display dataset dimensions
print("\n--- Dataset Shape ---")
print(df.shape)

# Display column information
print("\n--- Dataset Information ---")
df.info()

# Display missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Display target distribution
print("\n--- Target Distribution ---")
print(df[15].value_counts(dropna=False))
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def advanced_imputation(df, has_class_column=True):
    """
    Advanced imputation strategy for NDVI time series data
    Much better than simple mean imputation
    
    Parameters:
        df: DataFrame to impute
        has_class_column: Boolean indicating if df has a 'class' column (True for training, False for test)
    """
    print("Applying advanced imputation strategy...")
    df_imputed = df.copy()
    
    # Get NDVI columns
    ndvi_cols = [col for col in df.columns if col.endswith('_N')]
    print(f"Processing {len(ndvi_cols)} NDVI columns...")
    
    # Step 1: Temporal interpolation within each time series (row)
    print("Step 1: Temporal interpolation...")
    for idx in df_imputed.index:
        row_data = df_imputed.loc[idx, ndvi_cols]
        # Interpolate missing values using temporal relationships
        interpolated = row_data.interpolate(method='linear', limit_direction='both')
        df_imputed.loc[idx, ndvi_cols] = interpolated
    
    # Step 2: Class-wise seasonal imputation for remaining missing values
    if has_class_column:
        print("Step 2: Class-based seasonal imputation...")
        for class_name in df['class'].unique():
            class_mask = df['class'] == class_name
            class_data = df_imputed.loc[class_mask, ndvi_cols]
            
            # Calculate class-specific seasonal patterns (more robust with median)
            class_seasonal_patterns = class_data.median()
            
            # Fill remaining missing values with class-specific seasonal patterns
            for col in ndvi_cols:
                missing_in_class = df_imputed.loc[class_mask, col].isnull()
                if missing_in_class.any():
                    df_imputed.loc[class_mask & missing_in_class, col] = class_seasonal_patterns[col]
    else:
        print("Step 2: Global seasonal imputation (no class information available)...")
        # Use global patterns instead of class-specific ones
        global_patterns = df_imputed[ndvi_cols].median()
        for col in ndvi_cols:
            df_imputed[col].fillna(global_patterns[col], inplace=True)
    
    # Step 3: KNN imputation for any remaining stubborn missing values
    remaining_missing = df_imputed[ndvi_cols].isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Step 3: KNN imputation for {remaining_missing} remaining values...")
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_imputed[ndvi_cols] = knn_imputer.fit_transform(df_imputed[ndvi_cols])
    
    print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")
    return df_imputed

def denoise_time_series(series, window_length=5):
    """Apply Savitzky-Golay filter for denoising NDVI time series"""
    if len(series) < window_length or series.isnull().all():
        return series
    
    try:
        # Apply smoothing filter to reduce noise
        smoothed = savgol_filter(series, window_length, polyorder=2)
        return pd.Series(smoothed, index=series.index)
    except:
        return series

def create_advanced_features(df):
    """Create advanced features from NDVI time series"""
    print("Creating advanced features...")
    
    # Get NDVI columns
    ndvi_cols = [col for col in df.columns if col.endswith('_N')]
    X = df[ndvi_cols].copy()
    
    # Apply denoising
    print("Applying denoising...")
    for idx in X.index:
        X.loc[idx] = denoise_time_series(X.loc[idx])
    
    # Clip NDVI values to valid range
    X = X.clip(-1, 1)
    
    # Create feature matrix
    features = pd.DataFrame(index=X.index)
    
    # Basic statistical features
    features['ndvi_mean'] = X.mean(axis=1)
    features['ndvi_std'] = X.std(axis=1)
    features['ndvi_min'] = X.min(axis=1)
    features['ndvi_max'] = X.max(axis=1)
    features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']
    features['ndvi_median'] = X.median(axis=1)
    features['ndvi_q25'] = X.quantile(0.25, axis=1)
    features['ndvi_q75'] = X.quantile(0.75, axis=1)
    features['ndvi_iqr'] = features['ndvi_q75'] - features['ndvi_q25']
    
    # Robust statistics (less affected by noise)
    features['ndvi_robust_mean'] = X.apply(lambda row: stats.trim_mean(row.dropna(), 0.1), axis=1)
    features['ndvi_mad'] = X.apply(lambda row: stats.median_abs_deviation(row.dropna()), axis=1)
    
    # Temporal features
    features['ndvi_trend'] = X.apply(lambda row: calculate_trend(row), axis=1)
    features['ndvi_cv'] = features['ndvi_std'] / (features['ndvi_mean'].abs() + 1e-8)
    
    # Seasonal features (assuming columns are chronologically ordered)
    n_cols = len(ndvi_cols)
    
    # Growing season vs dormant season
    growing_season_cols = ndvi_cols[n_cols//4:3*n_cols//4]  # Middle 50% of year
    dormant_season_cols = ndvi_cols[:n_cols//4] + ndvi_cols[3*n_cols//4:]  # First and last 25%
    
    features['growing_season_mean'] = X[growing_season_cols].mean(axis=1)
    features['dormant_season_mean'] = X[dormant_season_cols].mean(axis=1)
    features['seasonal_amplitude'] = features['growing_season_mean'] - features['dormant_season_mean']
    
    # Peak analysis
    features['peak_count'] = X.apply(lambda row: count_peaks(row), axis=1)
    features['valley_count'] = X.apply(lambda row: count_valleys(row), axis=1)
    
    # Stability and variability
    features['stability_index'] = 1 / (1 + features['ndvi_std'])
    features['greenness_persistence'] = (X > 0.3).sum(axis=1) / len(ndvi_cols)
    
    # Growth dynamics
    X_diff = X.diff(axis=1)
    features['max_growth_rate'] = X_diff.max(axis=1)
    features['max_decline_rate'] = X_diff.min(axis=1)
    features['growth_volatility'] = X_diff.std(axis=1)
    
    # Percentile features
    features['ndvi_p10'] = X.quantile(0.1, axis=1)
    features['ndvi_p90'] = X.quantile(0.9, axis=1)
    
    # Add some key original NDVI values
    key_timepoints = [0, len(ndvi_cols)//4, len(ndvi_cols)//2, 3*len(ndvi_cols)//4, -1]
    for i, tp in enumerate(key_timepoints):
        if tp < len(ndvi_cols):
            features[f'ndvi_key_{i}'] = X.iloc[:, tp]
    
    print(f"Created {len(features.columns)} features")
    return features

def calculate_trend(series):
    """Calculate linear trend of time series"""
    valid_data = series.dropna()
    if len(valid_data) < 3:
        return 0
    x = np.arange(len(valid_data))
    try:
        slope, _, _, _, _ = stats.linregress(x, valid_data)
        return slope
    except:
        return 0

def count_peaks(series):
    """Count peaks in time series"""
    valid_data = series.dropna().values
    if len(valid_data) < 3:
        return 0
    peaks = 0
    for i in range(1, len(valid_data) - 1):
        if valid_data[i] > valid_data[i-1] and valid_data[i] > valid_data[i+1]:
            peaks += 1
    return peaks

def count_valleys(series):
    """Count valleys in time series"""
    valid_data = series.dropna().values
    if len(valid_data) < 3:
        return 0
    valleys = 0
    for i in range(1, len(valid_data) - 1):
        if valid_data[i] < valid_data[i-1] and valid_data[i] < valid_data[i+1]:
            valleys += 1
    return valleys

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("Loading data...")
# Load training data
df = pd.read_csv(r"assignment\assignment 2\hacktrain.csv")
test_data = pd.read_csv(r"assignment\assignment 2\hacktest.csv")

print(f"Training data shape: {df.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Classes: {df['class'].unique()}")
print(f"Class distribution:\n{df['class'].value_counts()}")

# Check missing values
print(f"\nMissing values in training data:\n{df.isnull().sum().sum()} total missing values")
print(f"Missing values in test data:\n{test_data.isnull().sum().sum()} total missing values")

# Apply advanced imputation to both datasets
print("\n=== PREPROCESSING TRAINING DATA ===")
df_imputed = advanced_imputation(df, has_class_column=True)

print("\n=== PREPROCESSING TEST DATA ===")
test_data_imputed = advanced_imputation(test_data, has_class_column=False)

# Create advanced features for training data
print("\n=== FEATURE ENGINEERING FOR TRAINING DATA ===")
X_features = create_advanced_features(df_imputed)

# Create advanced features for test data
print("\n=== FEATURE ENGINEERING FOR TEST DATA ===")
X_test_features = create_advanced_features(test_data_imputed)

# Prepare target variable
y = df_imputed['class'].copy()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFinal training features shape: {X_features.shape}")
print(f"Final test features shape: {X_test_features.shape}")

# Scale features (important for logistic regression)
print("\n=== SCALING FEATURES ===")
scaler = RobustScaler()  # RobustScaler is less affected by outliers
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_features), 
    columns=X_features.columns, 
    index=X_features.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_features), 
    columns=X_test_features.columns, 
    index=X_test_features.index
)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Train optimized logistic regression model
print("\n=== TRAINING MODEL ===")
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=2000,  # Increased iterations
    C=0.1,  # Regularization to handle noise
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train, y_train)

# Validate model
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Cross-validation for robust performance estimate
print("\n=== CROSS-VALIDATION ===")
cv_scores = cross_val_score(
    model, X_scaled, y_encoded, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Classification report
print("\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(
    y_val, y_val_pred,
    target_names=label_encoder.classes_,
    digits=4
))

# Train final model on all data
print("\n=== TRAINING FINAL MODEL ON ALL DATA ===")
final_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=2000,
    C=0.1,
    random_state=42,
    class_weight='balanced'
)
final_model.fit(X_scaled, y_encoded)

# Make predictions on test set
print("\n=== MAKING PREDICTIONS ON TEST SET ===")
test_predictions = final_model.predict(X_test_scaled)
test_predictions_decoded = label_encoder.inverse_transform(test_predictions)

# Create submission file
print("\n=== CREATING SUBMISSION FILE ===")
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'class': test_predictions_decoded
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Predicted class distribution:\n{submission['class'].value_counts()}")

print("\n=== PROCESS COMPLETE ===")
print("Key improvements made:")
print("1. Advanced temporal and class-based imputation")
print("2. Comprehensive feature engineering from NDVI time series")
print("3. Noise reduction with Savitzky-Golay filtering")
print("4. Robust scaling and regularization")
print("5. Cross-validation for model reliability")
print("6. Class balancing to handle imbalanced data")

from sklearn.metrics import accuracy_score


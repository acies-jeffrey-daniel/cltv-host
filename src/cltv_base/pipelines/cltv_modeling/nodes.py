import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error

def predict_cltv_bgf_ggf(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits BG/NBD and Gamma-Gamma models to predict 3-month CLTV (fixed horizon).
    This function acts as a Kedro node.
    """
    print(f"Predicting CLTV using BG/NBD + Gamma-Gamma models for 3 months (fixed horizon)...")
    df = transactions_df.copy()
    
    if not all(col in df.columns for col in ['Purchase Date', 'User ID', 'Total Amount']):
        raise KeyError("Missing required columns ('Purchase Date', 'User ID', 'Total Amount') in the transaction dataset for CLTV prediction.")

    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    observation_period_end = df['Purchase Date'].max()
    summary_df = summary_data_from_transaction_data(
        df, 
        customer_id_col='User ID',
        datetime_col='Purchase Date',
        monetary_value_col='Total Amount',
        observation_period_end=observation_period_end
    )

    summary_df = summary_df[(summary_df['frequency'] > 0) & (summary_df['monetary_value'] > 0)]
    if summary_df.empty:
        print("Warning: No valid data for CLTV prediction after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=['User ID', 'predicted_cltv_3m'])

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary_df['frequency'], summary_df['monetary_value'])

    summary_df['predicted_cltv_3m'] = ggf.customer_lifetime_value(
        bgf,
        summary_df['frequency'],
        summary_df['recency'],
        summary_df['T'],
        summary_df['monetary_value'],
        time=3,
        freq='D', 
        discount_rate=0.01
    )

    summary_df = summary_df.reset_index()
    summary_df['User ID'] = summary_df['User ID'].astype(str)
    
    print(f"Diagnostic: predict_cltv_bgf_ggf - predicted_cltv_df head:\n{summary_df.head()}")
    print(f"Diagnostic: predict_cltv_bgf_ggf - predicted_cltv_df null counts:\n{summary_df.isnull().sum()}")
    print(summary_df)
    return summary_df[['User ID', 'predicted_cltv_3m']]

def predict_cltv_xgboost(customers_df: pd.DataFrame, predicted_churn_probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Predict CLTV using XGBoost regression.
    Automatically adapts to user-uploaded data by selecting only available necessary features.
    Acts as a Kedro node.
    """
    print("Predicting CLTV using XGBoost...")

    df = customers_df.copy()

    df = df.merge(
        predicted_churn_probabilities[['User ID', 'predicted_churn_prob']],
        on='User ID',
        how='left'
    )

    df['predicted_churn_prob'] = df['predicted_churn_prob'].fillna(0)
    target_col = "Total Order Value"
    candidate_features = [
        'Total Unique Visits','Total Unique Sessions','Total Unique Devices',
        'Device Type','Channel','Geo Location',
        'Total Session Duration','Avg Session Duration','Total Page Views',
        'Avg Page Views','Total Bounces','Bounce Rate',
        'sessions_score','visits_score','pageviews_score','bounce_score','Engagement_Score', 'predicted_churn_prob'
    ]

    if target_col not in df.columns:
        raise KeyError(f"Missing required target column '{target_col}' in dataset.")

    available_features = [col for col in candidate_features if col in df.columns]
    print(f"Using features: {available_features}")

    X = df[available_features].copy()
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"XGBoost RMSE: {rmse}")

    df['cltv_xgboost'] = model.predict(X)
    return df[['User ID','cltv_xgboost']] if 'User ID' in df.columns else df[['cltv_xgboost']]

def predict_cltv_lightgbm(customers_df: pd.DataFrame, predicted_churn_probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Predict CLTV using LightGBM regression.
    Automatically adapts to user-uploaded data by selecting only available necessary features.
    Acts as a Kedro node.
    """
    print("Predicting CLTV using LightGBM...")

    df = customers_df.copy()
     # Assuming predicted_churn_probabilities has 'User ID' and 'predicted_churn_prob'
    df = df.merge(
        predicted_churn_probabilities[['User ID', 'predicted_churn_prob']],
        on='User ID',
        how='left'
    )
    df['predicted_churn_prob'] = df['predicted_churn_prob'].fillna(0)

    target_col = "Total Order Value"
    candidate_features = [
        'Total Unique Visits','Total Unique Sessions','Total Unique Devices',
        'Device Type','OS <lambda>','Channel','Geo Location',
        'Total Session Duration','Avg Session Duration','Total Page Views',
        'Avg Page Views','Total Bounces','Bounce Rate',
        'sessions_score','visits_score','pageviews_score','bounce_score','Engagement_Score', 'predicted_churn_prob'
    ]

    if target_col not in df.columns:
        raise KeyError(f"Missing required target column '{target_col}' in dataset.")

    available_features = [col for col in candidate_features if col in df.columns]
    print(f"Using features: {available_features}")

    X = df[available_features].copy()
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(objective='regression', n_estimators=300, learning_rate=0.05, max_depth=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"LightGBM RMSE: {rmse}")

    df['cltv_lightgbm'] = model.predict(X) 
    return df[['User ID','cltv_lightgbm']] if 'User ID' in df.columns else df[['cltv_lightgbm']]
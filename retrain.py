import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def retrain_model(
    new_data_path=r"./sales_data.csv", 
    old_data_path=r"./2.csv"
):
    old_df = pd.read_csv(old_data_path)
    new_df = pd.read_csv(new_data_path)

    for df in [old_df, new_df]:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df.to_csv(old_data_path, index=False)

    X = combined_df[['price', 'promotion', 'holiday', 'weather_index', 'day_of_week', 'month']]
    y = combined_df['units_sold']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return "Model retrained with combined data!"
if __name__ == "__main__":
    print(retrain_model())


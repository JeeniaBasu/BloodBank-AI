from utils.data_loader import load_data
from ml.stock_status_model import train_stock_status_model
from ml.demand_forecasting import train_demand_model

if __name__ == "__main__":
    frame = load_data()
    print("Training stock-status classifier...")
    classification = train_stock_status_model(frame)
    print(f"Accuracy: {classification['accuracy']:.4f}")
    print(f"Weighted F1: {classification['weighted_f1']:.4f}")
    print("Training demand forecasting model...")
    _, demand = train_demand_model(frame)
    print(f"MAE: {demand['mae']:.3f}")
    print(f"RMSE: {demand['rmse']:.3f}")
    print("Both models saved in the models/ directory.")

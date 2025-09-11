import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import ast
from tqdm import tqdm
from huggingface_hub import login
import importlib
import argparse
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, ETTh1Dataset, SunspotsDataset, ElectricityConsumptionZurichDataset,TaxiNewYorkDataset
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, Autoformer, FEDformer, Informer
import pytorch_lightning as pl

# Dataset configurations with proper frequencies
DATASET_CONFIGS = {
    "etth1": {
        "dataset_cls": ETTh1Dataset,
        "target_column": "OT",
        "forecast_horizon": 24,
        "size": None,
        "freq": 1  # Hourly frequency
    },
    "sunspots": {
        "dataset_cls": SunspotsDataset,
        "target_column": None,
        "forecast_horizon": 12,
        "size": None,
        "freq": 1  # Monthly frequency
    },
    "electricity": {
        "dataset_cls": ElectricityConsumptionZurichDataset,  # Using Darts' built-in dataset
        "target_column": "Value_NE5",
        "forecast_horizon": 48,
        "size": None,
        "freq": 1  # Numeric frequency (keep original resolution)
    },
    "taxi": {
        "dataset_cls": TaxiNewYorkDataset,  # Using Darts' built-in dataset
        "target_column": None,
        "forecast_horizon": 48,
        "size": None,
        "freq": 1  # Numeric frequency (keep original resolution)
    }
}

# Set device and CUDA fallback
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Configuration
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxx"
TEST_SPLIT = 0.2
OUTPUT_DIR = "results/ResultsModels"

# Model configurations with updated context lengths
MODEL_CONFIGS = {
    "PatchTST": {
        "model_class": "PatchTSTForPrediction",
        "config_class": "PatchTSTConfig",
        "model_id": "ibm-research/testing-patchtst_etth1_forecast",
        "context_length": 512,
        "prediction_length": 24,
        "model_params": {"num_input_channels": 1},
        "framework": "transformers"
    },
    "NHiTS": {
        "model_class": NHITS,
        "context_length": 512,
        "prediction_length": 24,
        "model_params": {},
        "framework": "neuralforecast"
    },
    "Autoformer": {
        "model_class": Autoformer,
        "context_length": 168,  # Increased for weekly seasonality
        "prediction_length": 24,
        "model_params": {},
        "framework": "neuralforecast"
    },
    "FEDformer": {
        "model_class": FEDformer,
        "context_length": 168,  # Increased for weekly seasonality
        "prediction_length": 24,
        "model_params": {},
        "framework": "neuralforecast"
    },
    "Informer": {
        "model_class": Informer,
        "context_length": 168,  # Increased for weekly seasonality
        "prediction_length": 24,
        "model_params": {},
        "framework": "neuralforecast"
    }
}

# Build dataset loaders dynamically
DATASET_LOADERS = {}
for ds_name, config in DATASET_CONFIGS.items():
    key = ds_name.lower()
    
    # All datasets now use Darts' built-in loaders
    DATASET_LOADERS[key] = lambda c=config: (
        c["dataset_cls"]()
        .load()
        .univariate_component(c["target_column"] or 0)
    )

# Update the prepare_nf_dataframe function to handle numeric frequency
def prepare_nf_dataframe(series, freq):
    """Create NeuralForecast dataframe with proper frequency handling"""
    if isinstance(freq, int):
        # For numeric frequency, use simple integer index
        return pd.DataFrame({
            "ds": np.arange(len(series)),
            "y": series,
            "unique_id": "series"
        })
    else:
        # For time-based frequency, use proper datetime index
        start_date = pd.Timestamp("2000-01-01")  # Generic start date
        dates = pd.date_range(start=start_date, periods=len(series), freq=freq)
        return pd.DataFrame({
            "ds": dates,
            "y": series,
            "unique_id": "series"
        })

# Custom dataset loader function with improved handling
def load_custom_dataset(url, time_col, freq, time_format=None, target_column=None):
    """Load custom dataset from URL with robust error handling"""
    try:
        # Download and cache dataset
        cache_path = f"./data_cache/{url.split('/')[-1]}"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if not os.path.exists(cache_path):
            df = pd.read_csv(url)
            df.to_csv(cache_path, index=False)
        else:
            df = pd.read_csv(cache_path)
        
        # Parse datetime with error handling
        try:
            if time_format:
                df[time_col] = pd.to_datetime(df[time_col], format=time_format)
            else:
                df[time_col] = pd.to_datetime(df[time_col])
        except ValueError as e:
            print(f"Date parsing error: {e}. Trying fallback parsing...")
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
        # Handle missing dates
        df = df.set_index(time_col).sort_index()
        if freq:
            df = df.asfreq(freq)
        
        # Create time series
        if target_column:
            ts = TimeSeries.from_series(df[target_column], freq=freq)
        else:
            ts = TimeSeries.from_series(df.iloc[:, 0], freq=freq)
        
        return ts
    except Exception as e:
        raise RuntimeError(f"Error loading custom dataset: {str(e)}")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed_and_device(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            # Try allocating a small tensor
            _ = torch.tensor([1.0], device='cuda')
            device = torch.device("cuda")
        except RuntimeError as e:
            print(f"CUDA error: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

def load_and_preprocess_data(dataset_name):
    """Load and preprocess data with improved handling"""
    
    try:
        ts = DATASET_LOADERS[dataset_name]()
        # Handle missing values
        #print(f"{dataset_name} has {ts.pd_series().head()} missing values")
        ts = ts.to_dataframe()
        cl = DATASET_CONFIGS[dataset_name]["target_column"]
        values = ts[cl].iloc[:].values.astype(np.float32) if cl else ts.iloc[:,0].values.astype(np.float32)
        
        # Ensure no NaNs remain
        if np.isnan(values).any():
            print(f"Warning: {np.isnan(values).sum()} NaNs found in {dataset_name}, filling with 0")
            values = np.nan_to_num(values)
            
        return values, DATASET_CONFIGS[dataset_name]["freq"]
    except Exception as e:
        raise RuntimeError(f"Error loading {dataset_name}: {str(e)}")

def prepare_nf_dataframe(series, freq):
    """Create NeuralForecast dataframe with proper frequency handling
    
    Args:
        series (np.array): Time series values
        freq: Frequency of the time series. Can be:
              - int (1, 2, etc.) for numeric frequencies
              - str ('H', 'D', etc.) for time-based frequencies
              - None for no frequency information
    
    Returns:
        pd.DataFrame: DataFrame in NeuralForecast format
    """
    if isinstance(freq, int) or freq is None:
        # For numeric frequency or no frequency, use simple integer index
        return pd.DataFrame({
            "ds": np.arange(len(series)),  # 0, 1, 2, ...
            "y": series,
            "unique_id": "series"
        })
    else:
        # For time-based frequency, use proper datetime index
        # Using a more recent default start date (2000 instead of 1800)
        start_date = pd.Timestamp("2000-01-01")  
        dates = pd.date_range(start=start_date, periods=len(series), freq=freq)
        return pd.DataFrame({
            "ds": dates,
            "y": series,
            "unique_id": "series"
        })

def load_model(model_name, prediction_length):
    """Load model with dynamic prediction length"""
    cfg = MODEL_CONFIGS[model_name].copy()
    cfg["prediction_length"] = prediction_length
    
    if cfg["framework"] == "transformers":
        try:
            config_module = "transformers.models.patchtst" if model_name == "PatchTST" else "transformers"
            model_module = config_module
            config_class = getattr(importlib.import_module(config_module), cfg["config_class"])
            model_class = getattr(importlib.import_module(model_module), cfg["model_class"])
            
            # Update config with prediction length
            model_params = cfg["model_params"].copy()
            model_params.update({
                "prediction_length": prediction_length,
                "context_length": cfg["context_length"]
            })
            
            config = config_class.from_pretrained(
                cfg["model_id"], 
                token=HF_TOKEN, 
                **model_params
            )
            model = model_class.from_pretrained(
                cfg["model_id"], 
                config=config, 
                ignore_mismatched_sizes=True, 
                token=HF_TOKEN
            )
            return model, cfg
        except Exception as e:
            raise RuntimeError(f"Error loading transformer model {model_name}: {str(e)}")
    else:
        try:
            model_class = cfg["model_class"]
            model = model_class(
                h=prediction_length,
                input_size=cfg["context_length"],
                **cfg["model_params"]
            )
            return model, cfg
        except Exception as e:
            raise RuntimeError(f"Error loading neuralforecast model {model_name}: {str(e)}")

def get_normalization_params(series):
    """Robust normalization with min-max scaling"""
    min_val, max_val = np.min(series), np.max(series)
    scale = max_val - min_val if (max_val - min_val) > 0 else 1.0
    return min_val, scale

def denormalize(data, min_val, scale):
    return data * scale + min_val

class TimeSeriesTestDataset(Dataset):
    """Improved dataset with proper test split handling"""
    def __init__(self, series, context_length, prediction_length, test_split=TEST_SPLIT):
        self.series = series
        self.context_length = context_length
        self.prediction_length = prediction_length
        total_length = len(series)
        
        # Calculate test indices
        test_size = int(total_length * test_split)
        self.test_start = total_length - test_size - context_length - prediction_length
        self.test_start = max(0, self.test_start)
        self.test_end = total_length - context_length - prediction_length + 1
        
    def __len__(self):
        return max(0, self.test_end - self.test_start)
    
    def __getitem__(self, idx):
        adjusted_idx = self.test_start + idx
        past = self.series[adjusted_idx:adjusted_idx + self.context_length]
        future = self.series[adjusted_idx + self.context_length:adjusted_idx + self.context_length + self.prediction_length]
        return torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)

def rolling_forecast(data, model, cfg, min_val, scale, freq):
    """Improved forecasting with frequency handling"""
    if cfg["framework"] == "transformers":
        series = data
        series_normalized = (series - min_val) / scale
        ds = TimeSeriesTestDataset(series_normalized, cfg["context_length"], cfg["prediction_length"])
        
        if len(ds) == 0:
            raise ValueError("Test dataset is empty. Check context_length and prediction_length.")
        
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        model.eval()
        model.to(device)
        preds, trues = [], []
        
        with torch.no_grad():
            for past, future in tqdm(loader, desc="Forecasting"):
                past = past.to(device).unsqueeze(-1)
                outputs = model(past_values=past)
                
                if hasattr(outputs, 'prediction_outputs'):
                    prediction = outputs.prediction_outputs.squeeze(-1).cpu().numpy()
                else:
                    prediction = outputs.last_hidden_state[:, -cfg["prediction_length"]:, 0].cpu().numpy()
                
                preds.append(prediction)
                trues.append(future.numpy())
        
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        return denormalize(preds, min_val, scale), denormalize(trues, min_val, scale)
    else:
        nf_data = data
        model = NeuralForecast(models=[model], freq=freq)
        model.fit(df=nf_data)
        forecasts = model.predict().reset_index()
        
        # Extract predictions and actuals
        preds = forecasts["yhat"].values
        actuals = nf_data["y"].values
        
        # Align predictions with actuals
        test_size = int(len(actuals) * TEST_SPLIT)
        actuals_test = actuals[-test_size:]
        preds_test = preds[-len(actuals_test):]
        
        # Reshape to (num_samples, prediction_length)
        num_samples = len(preds_test) // cfg["prediction_length"]
        preds_reshaped = preds_test[:num_samples * cfg["prediction_length"]].reshape(-1, cfg["prediction_length"])
        trues_reshaped = actuals_test[:num_samples * cfg["prediction_length"]].reshape(-1, cfg["prediction_length"])
        
        return preds_reshaped, trues_reshaped

def save_results(preds, trues, model_name, dataset_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}_results.csv")
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for h in range(preds.shape[1]):
            header.extend([f"pred{h+1}", f"true{h+1}"])
        writer.writerow(header)
        for i in range(preds.shape[0]):
            row = []
            for h in range(preds.shape[1]):
                row.extend([preds[i, h], trues[i, h]])
            writer.writerow(row)
    print(f"Results saved to {filename}")

def run_model_forecast(model_name, dataset_name):
    print(f"\n=== Running {model_name} on {dataset_name} ===")
    try:
        # Load data with frequency information
        series, freq = load_and_preprocess_data(dataset_name)
        nf_data = prepare_nf_dataframe(series, freq)
        
        # Get dataset-specific horizon
        forecast_horizon = DATASET_CONFIGS[dataset_name]["forecast_horizon"]
        
        total_length = len(series)
        test_size = int(total_length * TEST_SPLIT)
        print(f"Total points: {total_length}, Test size: {test_size}")
        print(f"Frequency: {freq}, Forecast horizon: {forecast_horizon}")
        
        # Load model with dataset-specific horizon
        model, cfg = load_model(model_name, forecast_horizon)
        
        # Get normalization parameters
        min_val, scale = get_normalization_params(series)
        
        # Prepare data based on framework
        data = series if cfg["framework"] == "transformers" else nf_data
        
        # Run forecasting
        preds, trues = rolling_forecast(data, model, cfg, min_val, scale, freq)
        
        print(f"Predictions shape: {preds.shape}, Actuals shape: {trues.shape}")
        print("Saving results...")
        save_results(preds, trues, model_name, dataset_name)
        print(f"{model_name} on {dataset_name} completed successfully!\n")
    except Exception as e:
        print(f"Error running {model_name} on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Experiment")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()), help="Model to run")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()), help="Dataset to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed and device
    device = set_seed_and_device(args.seed)
    
    # Login to Hugging Face Hub
    login(token=HF_TOKEN)
    print(DATASET_LOADERS.keys())
    # Run forecasting
    run_model_forecast(args.model, args.dataset)
    print("All experiments completed!")

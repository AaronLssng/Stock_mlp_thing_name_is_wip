import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import stock_data  # Your C library wrapper
from yfin import fetch_and_save_stock, load_stock_from_binary

# ============================================
# 1. CONFIGURATION
# ============================================

SYMBOL = "AAPL"  # Change this to any stock you have in data/
TIMEFRAME = "1d"
YEARS = 10

# ============================================
# 2. LOAD DATA FROM BINARY FILES
# ============================================


def load_stock_data(symbol, timeframe="1d"):
    """Load stock data from binary file"""
    filepath = f"data/{symbol}/{timeframe}.bin"

    if not os.path.exists(filepath):
        print(f"✗ Binary file not found: {filepath}")
        print(f"  Fetching {symbol} data first...")
        fetch_and_save_stock(symbol, timeframe=timeframe, years=YEARS)

        # Try loading again
        if not os.path.exists(filepath):
            print(f"✗ Failed to fetch {symbol}")
            return None

    try:
        # Use your existing load function
        df = load_stock_from_binary(symbol, timeframe)
        if df is not None and not df.empty:
            print(f"✓ Loaded {len(df)} bars for {symbol}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            return df
        else:
            print(f"✗ No data for {symbol}")
            return None
    except Exception as e:
        print(f"✗ Error loading {symbol}: {e}")
        return None


# ============================================
# 3. FEATURE ENGINEERING
# ============================================


def add_features(df):
    """Add technical indicators as features"""
    df = df.copy()

    # Price returns
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)

    # Price ratios
    df["High_Low_Ratio"] = (df["High"] / df["Low"] - 1) * 100
    df["Close_Open_Ratio"] = (df["Close"] / df["Open"] - 1) * 100

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()
        df[f"Price_to_MA_{window}"] = df["Close"] / df[f"MA_{window}"] - 1

    # Volatility
    for window in [5, 10, 20]:
        df[f"Volatility_{window}d"] = df["Return_1d"].rolling(window).std()

    # Volume features
    df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

    # ATR (Average True Range)
    df["True_Range"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["ATR_14"] = df["True_Range"].rolling(14).mean()

    # Drop NaN rows
    df = df.dropna()

    return df


# ============================================
# 4. LOAD AND PREPARE DATA
# ============================================

# Load stock data from binary
df = load_stock_data(SYMBOL, TIMEFRAME)

if df is None:
    print(f"✗ Failed to load {SYMBOL}. Exiting.")
    exit(1)

# Add features
print(f"\nAdding technical features...")
df = add_features(df)
print(f"Features added. Shape: {df.shape}")

# Prepare input and target
feature_cols = [
    col for col in df.columns if col not in ["Open", "High", "Low", "Close", "Volume"]
]
print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")

X_raw = df[feature_cols].values
target = df["Close"].pct_change().shift(-1).values  # Predict next day return

# Remove NaN rows
valid = ~np.isnan(target) & ~np.isnan(X_raw).any(axis=1)
X_raw = X_raw[valid]
target = target[valid]
dates = df.index.values[valid]

print(f"Clean data shape: X={X_raw.shape}, target={target.shape}")

# ============================================
# 5. TIME-SERIES SPLIT
# ============================================

split_ratio = 0.8
split_idx = int(len(X_raw) * split_ratio)

X_train_raw = X_raw[:split_idx]
X_test_raw = X_raw[split_idx:]
y_train_raw = target[:split_idx]
y_test_raw = target[split_idx:]
dates_train = dates[:split_idx]
dates_test = dates[split_idx:]

print(
    f"\nTrain period: {pd.Timestamp(dates_train[0])} to {pd.Timestamp(dates_train[-1])}"
)
print(f"Test period: {pd.Timestamp(dates_test[0])} to {pd.Timestamp(dates_test[-1])}")
print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

# Calculate class distribution
train_up = np.sum(y_train_raw > 0)
train_down = np.sum(y_train_raw < 0)
print(f"\nTraining class distribution:")
print(f"  Up days: {train_up} ({train_up / len(y_train_raw):.1%})")
print(f"  Down days: {train_down} ({train_down / len(y_train_raw):.1%})")

# ============================================
# 6. NORMALIZATION
# ============================================


def normalize_data(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, mean, std


X_train_norm, X_test_norm, mean, std = normalize_data(X_train_raw, X_test_raw)

# Normalize target
y_mean = y_train_raw.mean()
y_std = y_train_raw.std()

y_train_norm = (y_train_raw - y_mean) / y_std
y_test_norm = (y_test_raw - y_mean) / y_std

y_train = y_train_raw
y_test = y_test_raw

print(f"\nTraining stats: y_mean={y_mean:.6f}, y_std={y_std:.6f}")

# ============================================
# 7. MLP FUNCTIONS (BALANCED VERSION)
# ============================================


def ReLU(x):
    return np.maximum(0, x)


def ReLU_deriv(x):
    return np.where(x > 0, 1, 0)


def initialize_layer(input_size, output_size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    scale = np.sqrt(2.0 / input_size)
    weights = np.random.randn(input_size, output_size) * scale
    biases = np.zeros(output_size)
    return weights, biases


def initialize_network(layer_size, seed=42):
    network = []
    for i in range(len(layer_size) - 1):
        W, b = initialize_layer(
            layer_size[i], layer_size[i + 1], seed + i if seed else None
        )
        network.append({"W": W, "b": b, "Z": None, "A": None})
    return network


def forward_pass(network, X):
    activations = [X]
    for layer_idx, layer in enumerate(network):
        W = layer["W"]
        b = layer["b"]
        Z = np.dot(activations[-1], W) + b

        if layer_idx == len(network) - 1:
            A = Z
        else:
            A = ReLU(Z)

        layer["Z"] = Z
        layer["A"] = A
        activations.append(A)

    return activations, network


def weighted_mse(y_true, y_pred, down_weight=4.5):
    """Balanced weighted MSE"""
    weights = np.where(y_true < 0, down_weight, 1.0)
    weights = weights * (1.0 + np.abs(y_true) * 3)
    return np.mean(weights * (y_true - y_pred) ** 2)


def weighted_mse_derivative(y_true, y_pred, down_weight=4.5):
    weights = np.where(y_true < 0, down_weight, 1.0)
    weights = weights * (1.0 + np.abs(y_true) * 3)
    return 2 * weights * (y_pred - y_true) / len(y_true)


def backward_pass(network, activations, y_true, lambda_reg=0.0001, down_weight=4.5):
    gradients = []
    m = len(y_true)
    y_pred = activations[-1]

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    dA = weighted_mse_derivative(y_true, y_pred, down_weight)

    for layer_idx in range(len(network) - 1, -1, -1):
        layer = network[layer_idx]

        if layer_idx == len(network) - 1:
            dZ = dA
        else:
            dZ = dA * ReLU_deriv(layer["Z"])

        prev_activation = activations[layer_idx]

        if dZ.ndim == 1:
            dZ = dZ.reshape(-1, 1)

        dW = np.dot(prev_activation.T, dZ) / m + lambda_reg * layer["W"]
        db = np.sum(dZ, axis=0) / m

        dW = np.clip(dW, -1.0, 1.0)
        db = np.clip(db, -1.0, 1.0)

        gradients.append({"dW": dW, "db": db})
        dA = np.dot(dZ, layer["W"].T)

    gradients.reverse()
    return gradients


def update_weights(network, gradients, learning_rate):
    for layer, grad in zip(network, gradients):
        layer["W"] -= learning_rate * grad["dW"]
        layer["b"] -= learning_rate * grad["db"]


def predict(network, X):
    activations, _ = forward_pass(network, X)
    return activations[-1]


# ============================================
# 8. TRAINING WITH TARGETED BALANCING
# ============================================


def train_with_targeted_balancing(
    network,
    X,
    y,
    epochs,
    learning_rate,
    batch_size=64,
    lambda_reg=0.0001,
    patience=20,
    down_weight=4.5,
    target_down_ratio=0.4,
    verbose=True,
):
    """
    Training with targeted down ratio in batches to achieve desired balance
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples = len(X)
    best_loss = float("inf")
    patience_counter = 0
    losses = []
    val_losses = []

    # Validation split
    val_size = int(n_samples * 0.1)
    indices = np.random.permutation(n_samples)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    X_train_val = X[train_idx]
    y_train_val = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    current_lr = learning_rate

    # Identify indices
    up_indices = np.where(y_train_val.flatten() >= 0)[0]
    down_indices = np.where(y_train_val.flatten() < 0)[0]

    print(
        f"  Training up samples: {len(up_indices)}, down samples: {len(down_indices)}"
    )
    print(f"  Target down ratio in batches: {target_down_ratio:.1%}")

    for epoch in range(epochs):
        # Shuffle
        np.random.shuffle(up_indices)
        np.random.shuffle(down_indices)

        batch_loss = 0
        n_batches = 0

        # Calculate batch composition
        batch_size_down = int(batch_size * target_down_ratio)
        batch_size_up = batch_size - batch_size_down

        n_batches_total = max(
            len(up_indices) // batch_size_up, len(down_indices) // batch_size_down
        )

        for batch_idx in range(n_batches_total):
            # Sample from up and down indices
            up_start = (batch_idx * batch_size_up) % len(up_indices)
            down_start = (batch_idx * batch_size_down) % len(down_indices)

            up_batch = up_indices[
                up_start : min(up_start + batch_size_up, len(up_indices))
            ]
            down_batch = down_indices[
                down_start : min(down_start + batch_size_down, len(down_indices))
            ]

            # If we don't have enough, sample with replacement
            if len(up_batch) < batch_size_up:
                up_batch = np.random.choice(up_indices, batch_size_up, replace=True)
            if len(down_batch) < batch_size_down:
                down_batch = np.random.choice(
                    down_indices, batch_size_down, replace=True
                )

            batch_idx_all = np.concatenate([up_batch, down_batch])
            np.random.shuffle(batch_idx_all)

            X_batch = X_train_val[batch_idx_all]
            y_batch = y_train_val[batch_idx_all]

            activations, network = forward_pass(network, X_batch)
            y_pred = activations[-1]

            loss = weighted_mse(y_batch, y_pred, down_weight)
            batch_loss += loss
            n_batches += 1

            gradients = backward_pass(
                network, activations, y_batch, lambda_reg, down_weight
            )
            update_weights(network, gradients, current_lr)

        # Validation
        val_pred = predict(network, X_val)
        val_loss = weighted_mse(y_val, val_pred, down_weight)

        avg_loss = batch_loss / n_batches
        losses.append(avg_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        if epoch > 0 and val_loss > val_losses[-2]:
            current_lr *= 0.95

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_network = [
                {"W": layer["W"].copy(), "b": layer["b"].copy()} for layer in network
            ]
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            for layer, best_layer in zip(network, best_network):
                layer["W"] = best_layer["W"]
                layer["b"] = best_layer["b"]
            break

        if verbose and epoch % 20 == 0:
            train_pred = predict(network, X_train_val)
            train_pred_flat = train_pred.flatten()
            train_true_flat = y_train_val.flatten()

            train_acc = np.mean(np.sign(train_true_flat) == np.sign(train_pred_flat))
            down_mask = train_true_flat < 0
            up_mask = train_true_flat >= 0

            down_acc = (
                np.mean(
                    np.sign(train_true_flat[down_mask])
                    == np.sign(train_pred_flat[down_mask])
                )
                if np.any(down_mask)
                else 0
            )
            up_acc = (
                np.mean(
                    np.sign(train_true_flat[up_mask])
                    == np.sign(train_pred_flat[up_mask])
                )
                if np.any(up_mask)
                else 0
            )

            print(
                f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}, "
                f"Acc = {train_acc:.2%}, Up Acc = {up_acc:.2%}, Down Acc = {down_acc:.2%}, LR = {current_lr:.6f}"
            )

    return network, losses, val_losses


# ============================================
# 9. GRID SEARCH FOR OPTIMAL CONFIGURATION
# ============================================

input_size = X_train_norm.shape[1]

# Test different configurations
configs = [
    (4.0, 0.45),
    (4.5, 0.45),
    (5.0, 0.45),
]

best_architecture = [input_size, 64, 32, 16, 1]
results = []

for down_weight, target_down_ratio in configs:
    print(f"\n{'=' * 60}")
    print(
        f"Testing: down_weight={down_weight}, target_down_ratio={target_down_ratio:.1%}"
    )
    print(f"Architecture: {best_architecture}")
    print(f"{'=' * 60}")

    network = initialize_network(best_architecture, seed=42)
    total_params = sum(layer["W"].size + layer["b"].size for layer in network)
    print(f"Total parameters: {total_params:,}")

    epochs = 200
    learning_rate = 0.003
    batch_size = 64
    lambda_reg = 0.0001

    print("Starting training...")
    network, train_losses, val_losses = train_with_targeted_balancing(
        network,
        X_train_norm,
        y_train_norm,
        epochs,
        learning_rate,
        batch_size,
        lambda_reg,
        patience=25,
        down_weight=down_weight,
        target_down_ratio=target_down_ratio,
        verbose=True,
    )

    # Evaluate on test set
    y_pred_test_norm = predict(network, X_test_norm)
    y_pred_test = y_pred_test_norm * y_std + y_mean

    # Find optimal threshold
    thresholds = np.linspace(-0.003, 0.003, 31)
    best_threshold = 0.0
    best_balanced_acc = 0
    best_up_acc = 0
    best_down_acc = 0

    for threshold in thresholds:
        pred_direction = np.where(y_pred_test.flatten() > threshold, 1, -1)
        actual_direction = np.sign(y_test)

        correct_up = np.sum((actual_direction == 1) & (pred_direction == 1))
        total_up = np.sum(actual_direction == 1)
        correct_down = np.sum((actual_direction == -1) & (pred_direction == -1))
        total_down = np.sum(actual_direction == -1)

        up_acc = correct_up / total_up if total_up > 0 else 0
        down_acc = correct_down / total_down if total_down > 0 else 0
        balanced_acc = (up_acc + down_acc) / 2

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = threshold
            best_up_acc = up_acc
            best_down_acc = down_acc

    pred_direction = np.where(y_pred_test.flatten() > best_threshold, 1, -1)
    direction_acc = np.mean(np.sign(y_test) == pred_direction)

    n_down_pred = np.sum(pred_direction == -1)
    n_up_pred = np.sum(pred_direction == 1)

    print(f"\nResults:")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Directional Accuracy: {direction_acc:.2%}")
    print(f"  Balanced Accuracy: {best_balanced_acc:.2%}")
    print(f"  Up Accuracy: {best_up_acc:.2%}")
    print(f"  Down Accuracy: {best_down_acc:.2%}")
    print(f"  Predictions: UP={n_up_pred}, DOWN={n_down_pred}")

    results.append(
        {
            "down_weight": down_weight,
            "target_down_ratio": target_down_ratio,
            "threshold": best_threshold,
            "direction_acc": direction_acc,
            "balanced_acc": best_balanced_acc,
            "up_acc": best_up_acc,
            "down_acc": best_down_acc,
            "n_up_pred": n_up_pred,
            "n_down_pred": n_down_pred,
            "network": network,
            "y_pred_test": y_pred_test,
        }
    )

# ============================================
# 10. SELECT BEST MODEL
# ============================================

best_result = max(results, key=lambda x: x["balanced_acc"])
best_network = best_result["network"]
best_threshold = best_result["threshold"]
best_y_pred = best_result["y_pred_test"]

print(f"\n{'=' * 60}")
print(f"BEST MODEL FOUND FOR {SYMBOL}")
print(f"{'=' * 60}")
print(f"Down Weight: {best_result['down_weight']}")
print(f"Target Down Ratio: {best_result['target_down_ratio']:.1%}")
print(f"Threshold: {best_threshold:.4f}")
print(f"Directional Accuracy: {best_result['direction_acc']:.2%}")
print(f"Balanced Accuracy: {best_result['balanced_acc']:.2%}")
print(f"Up Accuracy: {best_result['up_acc']:.2%}")
print(f"Down Accuracy: {best_result['down_acc']:.2%}")
print(f"Predictions: UP={best_result['n_up_pred']}, DOWN={best_result['n_down_pred']}")

# ============================================
# 11. SAVE MODEL
# ============================================

with open(f"trained_model_{SYMBOL}.pkl", "wb") as f:
    pickle.dump(best_network, f)

scaler_params = {
    "mean": mean,
    "std": std,
    "y_mean": y_mean,
    "y_std": y_std,
    "feature_cols": feature_cols,
    "split_idx": split_idx,
    "train_start": dates_train[0],
    "train_end": dates_train[-1],
    "test_start": dates_test[0],
    "test_end": dates_test[-1],
    "threshold": best_threshold,
    "down_weight": best_result["down_weight"],
    "target_down_ratio": best_result["target_down_ratio"],
    "symbol": SYMBOL,
}
with open(f"scaler_params_{SYMBOL}.pkl", "wb") as f:
    pickle.dump(scaler_params, f)

print(f"\n✓ Model and parameters saved for {SYMBOL}!")

# ============================================
# 12. DETAILED EVALUATION
# ============================================

print(f"\n{'=' * 60}")
print(f"DETAILED EVALUATION FOR {SYMBOL}")
print(f"{'=' * 60}")

y_pred_test = best_y_pred
pred_direction = np.where(y_pred_test.flatten() > best_threshold, 1, -1)
actual_direction = np.sign(y_test)

print(f"\nThreshold: {best_threshold:.4f}")

print("\nSAMPLE PREDICTIONS (First 30 test samples):")
print("Date       | Actual   | Predicted | Pred Dir | Correct")
print("-" * 65)

for i in range(min(30, len(y_test))):
    date_val = dates_test[i]
    date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")

    actual = y_test[i]
    pred = y_pred_test[i][0]
    pred_dir = "UP" if pred > best_threshold else "DOWN"
    actual_dir = "UP" if actual > 0 else "DOWN"
    correct = "✓" if np.sign(actual) == np.sign(pred - best_threshold) else "✗"
    print(f"{date_str} | {actual:+.6f} | {pred:+.6f} | {pred_dir:4s} | {correct}")

# Confusion matrix
print("\nConfusion Matrix Summary:")
correct_up = np.sum((actual_direction == 1) & (pred_direction == 1))
total_up = np.sum(actual_direction == 1)
correct_down = np.sum((actual_direction == -1) & (pred_direction == -1))
total_down = np.sum(actual_direction == -1)

print(f"  True Up: {total_up}, Pred Up: {correct_up} ({correct_up / total_up:.1%})")
print(
    f"  True Down: {total_down}, Pred Down: {correct_down} ({correct_down / total_down:.1%})"
)

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE!")
print(f"{'=' * 60}")

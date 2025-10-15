import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import skew
import json
import optuna

MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"

DATA_LABELS = ["idle", "raise_left", "raise_right", "raise_both", "wave_left", "wave_right", 
               "wave_both", "circle_left", "circle_right", "circle_both", "clap", "jump"] # TBC
NUM_CLASSES = len(DATA_LABELS)
NUM_DATA = 6
WINDOW_SIZE = 20
NUM_SENSORS = 2

DATA = "Data2"
DATA_FOLDER_NAME = f"Dataset/{DATA}"
EXPORT_FOLDER_NAME = f"Export ({DATA})"

SEED = 42
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_DUMMY_PER_LABEL = 100 # For dummy data generation


# =========================
# Data generation / import
# =========================

def generate_dummy_data(data_file, label_file):
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    os.makedirs(os.path.dirname(label_file), exist_ok=True)

    with open(data_file, "w") as df, open(label_file, "w") as lf:
        for label_index, label in enumerate(DATA_LABELS):
            for _ in range(NUM_DUMMY_PER_LABEL):
                lf.write(str(label_index) + "\n")

                matrix = np.random.randint(1000 * label_index, 1000 * (label_index + 1), size=(WINDOW_SIZE * NUM_SENSORS, NUM_DATA))
                for i, row in enumerate(matrix):
                    # Compute sensor_id based on row index
                    sensor_id = (i % NUM_SENSORS) + 1
                    df.write(str(sensor_id) + " " + " ".join(map(str, row)) + "\n")
                df.write("\n")
    
    print(f"Generated data matrices in {data_file} and labels in {label_file}")


def extract_features(matrix):
    features = []
    for i in range(matrix.shape[1]): # Each axis
        axis = matrix[:, i]
        fft_axis = np.fft.fft(axis)
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.max(axis),
            np.min(axis),
            np.sqrt(np.mean(axis**2)),
            skew(axis),
            np.max(np.abs(fft_axis)),
            np.max(np.angle(fft_axis))
        ])
    return np.array(features, dtype=np.float32)


def preprocess(buckets):
    if MODEL_TYPE == "Simplified MLP":
        # Size = (NUM_FEATURES * NUM_DATA * NUM_SENSORS)
        matrix = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
    elif MODEL_TYPE == "MLP":
        # Size = (WINDOW_SIZE * NUM_DATA * NUM_SENSORS)
        matrix = np.concatenate(buckets, axis=0).ravel().astype(np.float32)
    elif MODEL_TYPE == "RNN":
        # Size = (WINDOW_SIZE * NUM_SENSORS, NUM_DATA)
        matrix = np.concatenate(buckets, axis=0).astype(np.float32)
    elif MODEL_TYPE == "CNN":
        # Size = (NUM_DATA, WINDOW_SIZE * NUM_SENSORS)
        matrix = np.concatenate(buckets, axis=0).T.astype(np.float32)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    return matrix


def import_data_with_id(data_file, label_file, lines_per_matrix):
    with open(label_file, "r") as f:
        labels_numeric = [int(line.strip()) for line in f if line.strip()]

    matrices = []
    buckets = [[] for _ in range(NUM_SENSORS)]

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of a matrix
                if any(buckets):
                    assert sum(len(bucket) for bucket in buckets) == lines_per_matrix, "Matrix line count mismatch"
                    matrices.append(preprocess(buckets))
                    buckets = [[] for _ in range(NUM_SENSORS)]
            else:
                line_values = [int(x) for x in line.split(" ")]
                device_id = line_values[0]
                sensor_values = line_values[1:]
                buckets[device_id - 1].append(sensor_values)
        # Add last matrix if file does not end with empty line
        if any(buckets):
            assert sum(len(bucket) for bucket in buckets) == lines_per_matrix, "Matrix line count mismatch"
            matrices.append(preprocess(buckets))

    # Check consistency
    assert len(matrices) == len(labels_numeric), "Number of matrices and labels mismatch"

    X_tensor = torch.tensor(np.array(matrices, dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(labels_numeric, dtype=np.int64), dtype=torch.long)

    return X_tensor, y_tensor



# =========================
# Export
# =========================

def fold_bn_into_conv(conv_layer, bn_layer):
    W = conv_layer.weight.detach().cpu().numpy() # Shape: [out_channels, in_channels, kernel]
    b = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
    
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta = bn_layer.bias.detach().cpu().numpy()
    mean = bn_layer.running_mean.detach().cpu().numpy()
    var = bn_layer.running_var.detach().cpu().numpy()
    eps = bn_layer.eps

    # Fold BN into Conv
    std = np.sqrt(var + eps) # Shape: [out_channels]
    W_folded = W * (gamma / std)[:, None, None] # Broadcast over in_channels and kernel
    b_folded = beta + (b - mean) * (gamma / std)

    return W_folded, b_folded


def export_model(model):
    os.makedirs(f"{EXPORT_FOLDER_NAME}/npy", exist_ok=True)

    if MODEL_TYPE == "CNN":
        # Fold BN into conv layers
        W1, b1 = fold_bn_into_conv(model.conv1, model.bn1)
        W2, b2 = fold_bn_into_conv(model.conv2, model.bn2)
        np.save(f"{EXPORT_FOLDER_NAME}/npy/conv1_weight.npy", W1)
        np.save(f"{EXPORT_FOLDER_NAME}/npy/conv1_bias.npy", b1)
        np.save(f"{EXPORT_FOLDER_NAME}/npy/conv2_weight.npy", W2)
        np.save(f"{EXPORT_FOLDER_NAME}/npy/conv2_bias.npy", b2)

        # Save FC layers
        np.save(f"{EXPORT_FOLDER_NAME}/npy/fc1_weight.npy", model.fc1.weight.detach().cpu().numpy())
        np.save(f"{EXPORT_FOLDER_NAME}/npy/fc1_bias.npy", model.fc1.bias.detach().cpu().numpy())
        np.save(f"{EXPORT_FOLDER_NAME}/npy/fc2_weight.npy", model.fc2.weight.detach().cpu().numpy())
        np.save(f"{EXPORT_FOLDER_NAME}/npy/fc2_bias.npy", model.fc2.bias.detach().cpu().numpy())

    else:
        for name, param in model.state_dict().items():
            arr = param.cpu().numpy()
            file_path = os.path.join(f"{EXPORT_FOLDER_NAME}/npy", f"{name}.npy")
            np.save(file_path, arr)


def generate_c_headers():
    os.makedirs(f"{EXPORT_FOLDER_NAME}/hls_headers", exist_ok=True)
    
    def write_header(array, var_name, filename):
        with open(filename, "w") as f:
            f.write(f"#ifndef {var_name.upper()}_H\n")
            f.write(f"#define {var_name.upper()}_H\n\n")
            f.write(f"static const float {var_name}[] = {{\n")
            flat = array.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val:.6f}f, ")
                if (i+1) % 8 == 0:
                    f.write("\n")
            f.write("\n};\n\n")
            f.write(f"#endif // {var_name.upper()}_H\n")
        print(f"Header saved: {filename}")

    for fname in os.listdir(f"{EXPORT_FOLDER_NAME}/npy"):
        if fname.endswith(".npy"):
            var_name = fname.replace(".npy", "").replace(".", "_")
            array = np.load(os.path.join(f"{EXPORT_FOLDER_NAME}/npy", fname))
            header_file = os.path.join(f"{EXPORT_FOLDER_NAME}/hls_headers", f"{var_name}.h")
            write_header(array, var_name, header_file)



# =========================
# Model Definitions
# =========================

# CNN Model
class ActionCNN(nn.Module):
    def __init__(self, num_channels, num_classes, sequence_length,
                 conv1_out=6, conv2_out=3, kernel_size_conv=3, pool_size=2,
                 fc1_neurons=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, conv1_out, kernel_size=kernel_size_conv, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size_conv, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.fc1 = nn.Linear(conv2_out * (sequence_length // pool_size), fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# RNN Model
class ActionRNN(nn.Module):
    def __init__(self, num_channels, num_classes, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, x):        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Take last timestep output
        out = self.dropout(out)
        out = self.fc(out) # Map to class scores
        return out


# MLP Model
class ActionMLP(nn.Module):
    def __init__(self, input_size, num_classes,
                 hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# MLP Model with summarised data
class SimplifiedMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# =========================
# Objective for Optuna
# =========================

def objective(trial, train_dataset, val_dataset, X_tensor_shape):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model-specific hyperparameters
    if MODEL_TYPE == "CNN":
        conv1_out = trial.suggest_categorical("conv1_out", [4, 6, 8])
        conv2_out = trial.suggest_categorical("conv2_out", [2, 3, 4])
        kernel_size_conv = trial.suggest_categorical("kernel_size_conv", [2, 3, 5])
        pool_size = trial.suggest_categorical("pool_size", [2, 3])
        fc1_neurons = trial.suggest_categorical("fc1_neurons", [32, 64, 128])
        model = ActionCNN(num_channels=X_tensor_shape[1], num_classes=NUM_CLASSES,
                          sequence_length=X_tensor_shape[2],
                          conv1_out=conv1_out, conv2_out=conv2_out,
                          kernel_size_conv=kernel_size_conv, pool_size=pool_size,
                          fc1_neurons=fc1_neurons, dropout=dropout)
    elif MODEL_TYPE == "RNN":
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        model = ActionRNN(num_channels=X_tensor_shape[2], num_classes=NUM_CLASSES,
                          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif MODEL_TYPE == "MLP":
        hidden1 = trial.suggest_categorical("hidden1", [128, 256, 512])
        hidden2 = trial.suggest_categorical("hidden2", [64, 128, 256])
        model = ActionMLP(input_size=X_tensor_shape[1], num_classes=NUM_CLASSES,
                          hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    elif MODEL_TYPE == "Simplified MLP":
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        model = SimplifiedMLP(input_size=X_tensor_shape[1], num_classes=NUM_CLASSES,
                              hidden_size=hidden_size, dropout=dropout)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train few epochs for tuning
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    return val_acc



def main():
    data_file = f"{DATA_FOLDER_NAME}/data.txt"
    label_file = f"{DATA_FOLDER_NAME}/label.txt"

    if not (os.path.exists(data_file) and os.path.exists(label_file)):
        should_generate_data = input("Generate dummy data? Y/N: ")
        if should_generate_data.upper() == "Y":
            generate_dummy_data(data_file, label_file)

    # Prepare data
    X_tensor, y_tensor = import_data_with_id(data_file, label_file, WINDOW_SIZE * NUM_SENSORS)
    dataset = TensorDataset(X_tensor, y_tensor)
    num_samples = len(dataset)

    # Split train, val, test
    labels = y_tensor.numpy()
    train_val_indices, test_indices = train_test_split(
        range(num_samples),
        test_size=0.15,
        stratify=labels,
        random_state=SEED
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.15 / 0.85,  # 15% of remaining for val
        stratify=labels[train_val_indices],
        random_state=SEED
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    # Optuna tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, X_tensor.shape), n_trials=20)
    print("Best hyperparameters:", study.best_params)

    # Retrain final model on train + val sets
    final_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    final_loader = DataLoader(final_train_dataset, batch_size=study.best_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=study.best_params["batch_size"], shuffle=False)

    # Construct final model using best hyperparameters
    best_params = study.best_params
    dropout = best_params["dropout"]
    if MODEL_TYPE == "CNN":
        model = ActionCNN(num_channels=X_tensor.shape[1], num_classes=NUM_CLASSES,
                          sequence_length=X_tensor.shape[2],
                          conv1_out=best_params["conv1_out"],
                          conv2_out=best_params["conv2_out"],
                          kernel_size_conv=best_params["kernel_size_conv"],
                          pool_size=best_params["pool_size"],
                          fc1_neurons=best_params["fc1_neurons"],
                          dropout=dropout)
    elif MODEL_TYPE == "RNN":
        model = ActionRNN(num_channels=X_tensor.shape[2], num_classes=NUM_CLASSES,
                          hidden_size=best_params["hidden_size"],
                          num_layers=best_params["num_layers"],
                          dropout=dropout)
    elif MODEL_TYPE == "MLP":
        model = ActionMLP(input_size=X_tensor.shape[1], num_classes=NUM_CLASSES,
                          hidden1=best_params["hidden1"],
                          hidden2=best_params["hidden2"],
                          dropout=dropout)
    elif MODEL_TYPE == "Simplified MLP":
        model = SimplifiedMLP(input_size=X_tensor.shape[1], num_classes=NUM_CLASSES,
                              hidden_size=best_params["hidden_size"], dropout=dropout)

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train final model
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in final_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/total:.4f}, Accuracy: {correct/total:.4f}")

    # Evaluate on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Export
    should_export = input("Export? Y/N: ")
    if should_export.upper() == "Y":
        os.makedirs(EXPORT_FOLDER_NAME, exist_ok=True)
        with open(f"{EXPORT_FOLDER_NAME}/best_params.txt", "w") as f:
            json.dump(best_params, f, indent=4)
        print("Best hyperparameters exported")

        export_model(model)
        generate_c_headers()
        torch.save(model, f"{EXPORT_FOLDER_NAME}/model.pt")
        print("Model exported")

        model.eval()
        with torch.no_grad():
            logits = model(X_tensor).numpy()
        np.savetxt(f"{EXPORT_FOLDER_NAME}/golden_logits.txt", logits, fmt="%.6f")
        print("Golden logits exported")


if __name__ == "__main__":
    main()

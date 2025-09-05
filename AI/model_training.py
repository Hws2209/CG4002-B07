import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import skew

MODEL_TYPE = "CNN"  # "CNN" | "RNN" | "MLP" | "Simplified MLP"

DATA_LABELS = ["class1", "class2", "class3", "class4"]
NUM_CLASSES = len(DATA_LABELS)
NUM_DATA = 6
DATA_FOLDER_NAME = "Dataset/DummyData"
EXPORT_FOLDER_NAME = "Export"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT = 0.3 # helps reduce overfitting
NUM_EPOCHS = 20

# For dummy data generation
SAMPLING_RATE = 20
TIME_LIMIT = 3
WINDOW_SIZE = SAMPLING_RATE * TIME_LIMIT
NUM_DUMMY_PER_LABEL = 100


def generate_dummy_data(data_file, label_file):
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    os.makedirs(os.path.dirname(label_file), exist_ok=True)

    with open(data_file, "w") as df, open(label_file, "w") as lf:
        for label_index, label in enumerate(DATA_LABELS):
            for _ in range(NUM_DUMMY_PER_LABEL):
                lf.write(str(label_index) + "\n")

                matrix = np.random.randint(1000 * label_index, 1000 * (label_index + 1), size=(WINDOW_SIZE, NUM_DATA))
                for row in matrix:
                    df.write(", ".join(map(str, row)) + "\n")
                df.write("\n")
    
    print(f"Generated data matrices in {data_file} and labels in {label_file}")


def extract_features(matrix):
    features = []
    for i in range(matrix.shape[1]):  # each axis
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


def import_data(data_file, label_file, lines_per_matrix):
    with open(label_file, "r") as f:
        labels_numeric = [int(line.strip()) for line in f if line.strip()]

    matrices = []
    current_matrix = []

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # empty line indicates end of a matrix
                if current_matrix:
                    matrices.append(np.array(current_matrix, dtype=float))
                    current_matrix = []
            else:
                current_matrix.append([float(x) for x in line.split(",")])
        # Add last matrix if file does not end with empty line
        if current_matrix:
            matrices.append(np.array(current_matrix, dtype=float))

    # Check consistency
    assert all(m.shape[0] == lines_per_matrix for m in matrices), "Matrix line count mismatch"
    assert len(matrices) == len(labels_numeric), "Number of matrices and labels mismatch"

    if MODEL_TYPE == "Simplified MLP":
        # Summarise data
        X_np = np.array([extract_features(m) for m in matrices], dtype=np.float32)
    elif MODEL_TYPE == "MLP":
        # Flatten each matrix
        X_np = np.array([m.flatten() for m in matrices], dtype=np.float32)
    elif MODEL_TYPE == "RNN":
        # Stack matrices into 3D array: (num_samples, sequence_length, num_channels)
        X_np = np.array([m for m in matrices], dtype=np.float32)
    elif MODEL_TYPE == "CNN":
        # Stack matrices into 3D array: (num_samples, num_channels, sequence_length)
        X_np = np.array([m.T for m in matrices], dtype=np.float32)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    y_np = np.array(labels_numeric, dtype=np.int64)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    return X_tensor, y_tensor


def fold_bn_into_conv(conv_layer, bn_layer):
    W = conv_layer.weight.detach().cpu().numpy()  # shape: [out_channels, in_channels, kernel]
    b = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
    
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta = bn_layer.bias.detach().cpu().numpy()
    mean = bn_layer.running_mean.detach().cpu().numpy()
    var = bn_layer.running_var.detach().cpu().numpy()
    eps = bn_layer.eps

    # Fold BN into Conv
    std = np.sqrt(var + eps)             # shape: [out_channels]
    W_folded = W * (gamma / std)[:, None, None]  # broadcast over in_channels and kernel
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
            f.write(f"float {var_name}[] = {{\n")
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


# CNN Model
class ActionCNN(nn.Module):
    def __init__(self, num_channels, num_classes, sequence_length):
        super(ActionCNN, self).__init__()

        conv1_out = 6
        conv2_out = 3
        kernel_size_conv = 3
        pool_size = 2
        fc1_neurons = 64
        
        self.conv1 = nn.Conv1d(num_channels, conv1_out, kernel_size=kernel_size_conv, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size_conv, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.fc1 = nn.Linear(conv2_out * (sequence_length // pool_size), fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# RNN Model
class ActionRNN(nn.Module):
    def __init__(self, num_channels, num_classes, sequence_length):
        super(ActionRNN, self).__init__()
        
        self.hidden_size = 64
        self.num_layers = 1
        
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=DROPOUT)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, x):        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # take last timestep output
        out = self.dropout(out)
        out = self.fc(out)   # map to class scores
        return out


# MLP Model
class ActionMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ActionMLP, self).__init__()

        hidden1 = 256
        hidden2 = 128

        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# MLP Model with summarised data
class SimplifiedMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimplifiedMLP, self).__init__()

        hidden_size=64

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    should_generate_data = input("Generate dummy data? Y/N: ")
    if should_generate_data.upper() == "Y":
        generate_dummy_data(f"{DATA_FOLDER_NAME}/data.txt", f"{DATA_FOLDER_NAME}/label.txt")

    # Prepare data
    X_tensor, y_tensor = import_data(f"{DATA_FOLDER_NAME}/data.txt", f"{DATA_FOLDER_NAME}/label.txt", WINDOW_SIZE)
    dataset = TensorDataset(X_tensor, y_tensor)
    num_samples = len(dataset)
    test_size = int(0.25 * num_samples)
    train_size = num_samples - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if MODEL_TYPE == "CNN":
        model = ActionCNN(num_channels=X_tensor.shape[1], num_classes=NUM_CLASSES, sequence_length=X_tensor.shape[2])
    elif MODEL_TYPE == "RNN":
        model = ActionRNN(num_channels=X_tensor.shape[2], num_classes=NUM_CLASSES, sequence_length=X_tensor.shape[1])
    elif MODEL_TYPE == "MLP":
        model = ActionMLP(input_size=X_tensor.shape[1], num_classes=NUM_CLASSES)
    elif MODEL_TYPE == "Simplified MLP":
        model = SimplifiedMLP(input_size=X_tensor.shape[1], num_classes=NUM_CLASSES)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # Evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("CNN Action Classification Confusion Matrix")
    plt.show()

    # Export params and model
    should_export_model = input("Export params and model? Y/N: ")
    if should_export_model.upper() == "Y":
        export_model(model)
        generate_c_headers()
        torch.save(model, f"{EXPORT_FOLDER_NAME}/model.pt")

    # Export golden logits
    should_export_test_vectors = input("Export golden logits? Y/N: ")
    if should_export_test_vectors.upper() == "Y":
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor).numpy()

        np.savetxt(f"{EXPORT_FOLDER_NAME}/golden_logits.txt", logits, fmt="%.6f")
        print("Golden logits exported")


if __name__ == "__main__":
    main()

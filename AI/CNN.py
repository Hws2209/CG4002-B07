import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


DATA_LABELS = ["logout", "shield", "reload", "grenade"]
NUM_CLASSES = len(DATA_LABELS)
NUM_DATA = 6
DATA_FOLDER_NAME = "Dataset/DummyData"
EXPORT_FOLDER_NAME = "Export"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
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


def import_data(data_file, label_file, lines_per_matrix):
    with open(label_file, "r") as f:
        labels_numeric = [int(line.strip()) for line in f if line.strip()]
        labels_one_hot = tf.keras.utils.to_categorical(labels_numeric, num_classes=NUM_CLASSES)

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
    assert len(matrices) == len(labels_one_hot), "Number of matrices and labels mismatch"

    # Stack matrices into 3D array: (num_samples, num_channels, sequence_length)
    X_np = np.array([m.T for m in matrices], dtype=np.float32)
    y_np = np.array([np.argmax(l) for l in labels_one_hot], dtype=np.int64)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    return X_tensor, y_tensor


def export_model(model, frac_bits=15):
    os.makedirs(f"{EXPORT_FOLDER_NAME}/npy", exist_ok=True)

    def float_to_fixed(arr, frac_bits=15):
        # Scale float32 to int16 for fixed-point
        scaled = np.round(arr * (2**frac_bits)).astype(np.int16)
        return scaled

    for name, param in model.state_dict().items():
        arr = param.cpu().numpy()
        fixed_arr = float_to_fixed(arr, frac_bits)
        file_path = os.path.join(f"{EXPORT_FOLDER_NAME}/npy", f"{name}.npy")
        np.save(file_path, fixed_arr)
        print(f"Exported {name} -> {file_path}")


def generate_c_headers():
    os.makedirs(f"{EXPORT_FOLDER_NAME}/hls_headers", exist_ok=True)
    
    def write_header(array, var_name, filename):
        with open(filename, "w") as f:
            f.write(f"#ifndef {var_name.upper()}_H\n")
            f.write(f"#define {var_name.upper()}_H\n\n")
            f.write(f"short {var_name}[] = {{\n")
            flat = array.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val}, ")
                if (i+1) % 16 == 0:
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
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * (sequence_length // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1) # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    should_generate_data = input("Generate dummy data? Y/N: ")
    if should_generate_data.upper() == "Y":
        generate_dummy_data(f"{DATA_FOLDER_NAME}/data.txt", f"{DATA_FOLDER_NAME}/label.txt")

    X_tensor, y_tensor = import_data(f"{DATA_FOLDER_NAME}/data.txt", f"{DATA_FOLDER_NAME}/label.txt", WINDOW_SIZE)
    dataset = TensorDataset(X_tensor, y_tensor)
    num_samples = len(dataset)
    test_size = int(0.25 * num_samples)
    train_size = num_samples - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sequence_length = X_tensor.shape[2]
    num_channels = X_tensor.shape[1]

    model = ActionCNN(num_channels=num_channels, num_classes=NUM_CLASSES, sequence_length=sequence_length)

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


if __name__ == "__main__":
    main()

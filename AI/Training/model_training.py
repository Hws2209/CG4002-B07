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

NUM_DATA = 6
WINDOW_SIZE = 20

MODE = 1
if MODE == 1:
    NUM_SENSORS = 2
    NUM_CLASSES = 12
else:
    NUM_SENSORS = 4
    NUM_CLASSES = 7


DATA = "Data6"
DATA_FOLDER_NAME = f"Dataset/{DATA}"
EXPORT_FOLDER_NAME = f"Export ({DATA})"

SEED = 42
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_DUMMY_PER_LABEL = 100 # For dummy data generation


# =========================
# Data generation / import
# =========================

def generate_dummy_data(dataFile, labelFile):
    os.makedirs(os.path.dirname(dataFile), exist_ok=True)
    os.makedirs(os.path.dirname(labelFile), exist_ok=True)

    with open(dataFile, "w") as df, open(labelFile, "w") as lf:
        for labelIndex in range(NUM_CLASSES):
            for _ in range(NUM_DUMMY_PER_LABEL):
                lf.write(str(labelIndex) + "\n")

                matrix = np.random.randint(1000 * labelIndex, 1000 * (labelIndex + 1), size=(WINDOW_SIZE * NUM_SENSORS, NUM_DATA))
                for i, row in enumerate(matrix):
                    # Compute deviceID based on row index
                    deviceID = (i % NUM_SENSORS) + 1
                    df.write(str(deviceID) + " " + " ".join(map(str, row)) + "\n")
                df.write("\n")
    
    print(f"Generated data matrices in {dataFile} and labels in {labelFile}")


def extract_features(matrix):
    features = []
    for i in range(matrix.shape[1]): # Each axis
        axis = matrix[:, i]
        fftAxis = np.fft.fft(axis)
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.max(axis),
            np.min(axis),
            np.sqrt(np.mean(axis**2)),
            skew(axis),
            np.max(np.abs(fftAxis)),
            np.max(np.angle(fftAxis))
        ])
    return np.array(features, dtype=np.float32)


def preprocess(buckets):
    if MODEL_TYPE == "Simplified MLP":
        # Size = (NUM_FEATURES * NUM_DATA * 2)
        matrix = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
    elif MODEL_TYPE == "MLP":
        # Size = (WINDOW_SIZE * NUM_DATA * 2)
        matrix = np.concatenate(buckets, axis=0).ravel().astype(np.float32)
    elif MODEL_TYPE == "RNN":
        # Size = (WINDOW_SIZE * 2, NUM_DATA)
        matrix = np.concatenate(buckets, axis=0).astype(np.float32)
    elif MODEL_TYPE == "CNN":
        # Size = (NUM_DATA, WINDOW_SIZE * 2)
        matrix = np.concatenate(buckets, axis=0).T.astype(np.float32)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    return matrix


def import_data_with_id(dataFile, labelFile, linesPerMatrix):
    with open(labelFile, "r") as f:
        labelsNumeric = [int(line.strip()) for line in f if line.strip()]

    matrices = []
    buckets = [[] for _ in range(NUM_SENSORS)]

    with open(dataFile, "r") as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of a matrix
                if any(buckets):
                    assert sum(len(bucket) for bucket in buckets) == linesPerMatrix, "Matrix line count mismatch"
                    if MODE == 1:
                        matrices.append(preprocess(buckets))
                    else:
                        matrices.append(preprocess(buckets[:2]))
                        matrices.append(preprocess(buckets[2:]))
                    buckets = [[] for _ in range(NUM_SENSORS)]
            else:
                lineValues = [int(x) for x in line.split(" ")]
                deviceID = lineValues[0]
                sensorValues = lineValues[1:]
                buckets[deviceID - 1].append(sensorValues)

        # Add last matrix if file does not end with empty line
        if any(buckets):
            assert sum(len(bucket) for bucket in buckets) == linesPerMatrix, "Matrix line count mismatch"
            if MODE == 1:
                matrices.append(preprocess(buckets))
            else:
                matrices.append(preprocess(buckets[:2]))
                matrices.append(preprocess(buckets[2:]))

    if MODE == 2:
        labelsNumeric = [label for label in labelsNumeric for _ in range(2)]
    
    # Check consistency
    assert len(matrices) == len(labelsNumeric), "Number of matrices and labels mismatch"

    XTensor = torch.tensor(np.array(matrices, dtype=np.float32), dtype=torch.float32)
    yTensor = torch.tensor(np.array(labelsNumeric, dtype=np.int64), dtype=torch.long)

    return XTensor, yTensor



# =========================
# Export
# =========================

def fold_bn_into_conv(convLayer, bnLayer):
    weights = convLayer.weight.detach().cpu().numpy() # Shape: [out_channels, in_channels, kernel]
    biases = convLayer.bias.detach().cpu().numpy() if convLayer.bias is not None else np.zeros(weights.shape[0], dtype=np.float32)
    
    gamma = bnLayer.weight.detach().cpu().numpy()
    beta = bnLayer.bias.detach().cpu().numpy()
    mean = bnLayer.running_mean.detach().cpu().numpy()
    var = bnLayer.running_var.detach().cpu().numpy()
    eps = bnLayer.eps

    # Fold BN into Conv
    std = np.sqrt(var + eps) # Shape: [out_channels]
    weightsFolded = weights * (gamma / std)[:, None, None] # Broadcast over in_channels and kernel
    biasesFolded = beta + (biases - mean) * (gamma / std)

    return weightsFolded, biasesFolded


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
            filePath = os.path.join(f"{EXPORT_FOLDER_NAME}/npy", f"{name}.npy")
            np.save(filePath, arr)


def generate_c_headers():
    os.makedirs(f"{EXPORT_FOLDER_NAME}/hls_headers", exist_ok=True)
    
    def write_header(array, varName, filename):
        with open(filename, "w") as f:
            f.write(f"#ifndef {varName.upper()}_H\n")
            f.write(f"#define {varName.upper()}_H\n\n")
            f.write(f"static const float {varName}[] = {{\n")
            flat = array.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val}f, ")
                if (i+1) % 8 == 0:
                    f.write("\n")
            f.write("\n};\n\n")
            f.write(f"#endif // {varName.upper()}_H\n")
        print(f"Header saved: {filename}")

    for fname in os.listdir(f"{EXPORT_FOLDER_NAME}/npy"):
        if fname.endswith(".npy"):
            varName = fname.replace(".npy", "").replace(".", "_")
            array = np.load(os.path.join(f"{EXPORT_FOLDER_NAME}/npy", fname))
            headerFile = os.path.join(f"{EXPORT_FOLDER_NAME}/hls_headers", f"{varName}.h")
            write_header(array, varName, headerFile)



# =========================
# Model Definitions
# =========================

# CNN Model
class ActionCNN(nn.Module):
    def __init__(self, numChannels, numClasses, sequenceLength,
                 conv1Out=6, conv2Out=3, kernelSizeConv=3, poolSize=2,
                 fc1Neurons=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(numChannels, conv1Out, kernel_size=kernelSizeConv, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1Out)
        self.conv2 = nn.Conv1d(conv1Out, conv2Out, kernel_size=kernelSizeConv, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2Out)
        self.pool = nn.MaxPool1d(kernel_size=poolSize)
        self.fc1 = nn.Linear(conv2Out * (sequenceLength // poolSize), fc1Neurons)
        self.fc2 = nn.Linear(fc1Neurons, numClasses)
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
    def __init__(self, numChannels, numClasses, hiddenSize=64, numLayers=1, dropout=0.3):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.lstm = nn.LSTM(input_size=numChannels, hidden_size=self.hiddenSize,
                            num_layers=self.numLayers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hiddenSize, numClasses)
    
    def forward(self, x):        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        c0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Take last timestep output
        out = self.dropout(out)
        out = self.fc(out) # Map to class scores
        return out


# MLP Model
class ActionMLP(nn.Module):
    def __init__(self, inputSize, numClasses,
                 hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, numClasses)
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
    def __init__(self, inputSize, numClasses, hiddenSize=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, numClasses)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# =========================
# Objective for Optuna
# =========================

def objective(trial, trainDataset, valDataset, XTensorShape):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batchSize = trial.suggest_categorical("batchSize", [16, 32, 64])
    
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

    # Model-specific hyperparameters
    if MODEL_TYPE == "CNN":
        conv1Out = trial.suggest_categorical("conv1Out", [4, 6, 8])
        conv2Out = trial.suggest_categorical("conv2Out", [2, 3, 4])
        kernelSizeConv = trial.suggest_categorical("kernelSizeConv", [2, 3])
        poolSize = trial.suggest_categorical("poolSize", [2, 4])
        fc1Neurons = trial.suggest_categorical("fc1Neurons", [64, 128])
        model = ActionCNN(numChannels=XTensorShape[1], numClasses=NUM_CLASSES,
                          sequenceLength=XTensorShape[2],
                          conv1Out=conv1Out, conv2Out=conv2Out,
                          kernelSizeConv=kernelSizeConv, poolSize=poolSize,
                          fc1Neurons=fc1Neurons, dropout=dropout)
    elif MODEL_TYPE == "RNN":
        hiddenSize = trial.suggest_categorical("hiddenSize", [32, 64, 128])
        numLayers = trial.suggest_categorical("numLayers", [1, 2, 3])
        model = ActionRNN(numChannels=XTensorShape[2], numClasses=NUM_CLASSES,
                          hiddenSize=hiddenSize, numLayers=numLayers, dropout=dropout)
    elif MODEL_TYPE == "MLP":
        hidden1 = trial.suggest_categorical("hidden1", [128, 256, 512])
        hidden2 = trial.suggest_categorical("hidden2", [64, 128, 256])
        model = ActionMLP(inputSize=XTensorShape[1], numClasses=NUM_CLASSES,
                          hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    elif MODEL_TYPE == "Simplified MLP":
        hiddenSize = trial.suggest_categorical("hiddenSize", [32, 64, 128])
        model = SimplifiedMLP(inputSize=XTensorShape[1], numClasses=NUM_CLASSES,
                              hiddenSize=hiddenSize, dropout=dropout)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train few epochs for tuning
    for epoch in range(5):
        model.train()
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in valLoader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valAcc = correct / total
    return valAcc



def main():
    dataFile = f"{DATA_FOLDER_NAME}/data.txt"
    labelFile = f"{DATA_FOLDER_NAME}/label.txt"

    if not (os.path.exists(dataFile) and os.path.exists(labelFile)):
        shouldGenerateData = input("Generate dummy data? Y/N: ")
        if shouldGenerateData.upper() == "Y":
            generate_dummy_data(dataFile, labelFile)

    # Prepare data
    XTensor, yTensor = import_data_with_id(dataFile, labelFile, WINDOW_SIZE * NUM_SENSORS)
    dataset = TensorDataset(XTensor, yTensor)
    numSamples = len(dataset)

    # Split train, val, test
    labels = yTensor.numpy()
    trainValIndices, testIndices = train_test_split(
        range(numSamples),
        test_size=0.15,
        stratify=labels,
        random_state=SEED
    )
    trainIndices, valIndices = train_test_split(
        trainValIndices,
        test_size=0.15 / 0.85,  # 15% of remaining for val
        stratify=labels[trainValIndices],
        random_state=SEED
    )
    trainDataset = Subset(dataset, trainIndices)
    valDataset   = Subset(dataset, valIndices)
    testDataset  = Subset(dataset, testIndices)

    # Optuna tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, trainDataset, valDataset, XTensor.shape), n_trials=20)
    print("Best hyperparameters:", study.best_params)

    # Retrain final model on train + val sets
    finalTrainDataset = torch.utils.data.ConcatDataset([trainDataset, valDataset])
    finalLoader = DataLoader(finalTrainDataset, batch_size=study.best_params["batchSize"], shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=study.best_params["batchSize"], shuffle=False)

    # Construct final model using best hyperparameters
    bestParams = study.best_params
    dropout = bestParams["dropout"]
    if MODEL_TYPE == "CNN":
        model = ActionCNN(numChannels=XTensor.shape[1], numClasses=NUM_CLASSES,
                          sequenceLength=XTensor.shape[2],
                          conv1Out=bestParams["conv1Out"],
                          conv2Out=bestParams["conv2Out"],
                          kernelSizeConv=bestParams["kernelSizeConv"],
                          poolSize=bestParams["poolSize"],
                          fc1Neurons=bestParams["fc1Neurons"],
                          dropout=dropout)
    elif MODEL_TYPE == "RNN":
        model = ActionRNN(numChannels=XTensor.shape[2], numClasses=NUM_CLASSES,
                          hiddenSize=bestParams["hiddenSize"],
                          numLayers=bestParams["numLayers"],
                          dropout=dropout)
    elif MODEL_TYPE == "MLP":
        model = ActionMLP(inputSize=XTensor.shape[1], numClasses=NUM_CLASSES,
                          hidden1=bestParams["hidden1"],
                          hidden2=bestParams["hidden2"],
                          dropout=dropout)
    elif MODEL_TYPE == "Simplified MLP":
        model = SimplifiedMLP(inputSize=XTensor.shape[1], numClasses=NUM_CLASSES,
                              hiddenSize=bestParams["hiddenSize"], dropout=dropout)

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=bestParams["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train final model
    for epoch in range(NUM_EPOCHS):
        model.train()
        runningLoss, correct, total = 0.0, 0, 0
        for inputs, labels in finalLoader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {runningLoss/total:.4f}, Accuracy: {correct/total:.4f}")

    # Evaluate on test set
    model.eval()
    allPreds, allLabels = [], []
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            allPreds.extend(predicted.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(allLabels, allPreds)
    print(cm)

    # Export
    shouldExport = input("Export? Y/N: ")
    if shouldExport.upper() == "Y":
        os.makedirs(EXPORT_FOLDER_NAME, exist_ok=True)
        with open(f"{EXPORT_FOLDER_NAME}/best_params.txt", "w") as f:
            json.dump(bestParams, f, indent=4)
        print("Best hyperparameters exported")

        export_model(model)
        generate_c_headers()
        torch.save(model, f"{EXPORT_FOLDER_NAME}/model.pt")
        print("Model exported")

        model.eval()
        with torch.no_grad():
            logits = model(XTensor).numpy()
        np.savetxt(f"{EXPORT_FOLDER_NAME}/golden_logits.txt", logits, fmt="%.6f")
        print("Golden logits exported")


if __name__ == "__main__":
    main()

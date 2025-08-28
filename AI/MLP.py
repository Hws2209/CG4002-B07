import os
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns


NEURONS_HIDDEN_LAYER = [56]
DROPOUT = 0.40
LEARNING_RATE = 0.01
EPOCHS = 10

NUM_FEATURES = 8
NUM_DATA = 6
NUM_INPUTS = NUM_FEATURES * NUM_DATA
DATA_LABELS = ["logout", "shield", "reload", "grenade"]
NUM_CLASSES = len(DATA_LABELS)
DATA_FOLDER_NAME = "Dataset/DummyData"
EXPORT_FOLDER_NAME = "Export"

# For dummy data generation
SAMPLING_RATE = 20
TIME_LIMIT = 3
WINDOW_SIZE = SAMPLING_RATE * TIME_LIMIT
NUM_DUMMY_PER_LABEL = 100



def get_model():
    # MLP model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(NEURONS_HIDDEN_LAYER[0], input_shape=(NUM_INPUTS,), activation="relu"))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    
    i = 1
    while (i < len(NEURONS_HIDDEN_LAYER)):
        model.add(tf.keras.layers.Dense(NEURONS_HIDDEN_LAYER[i], activation="relu"))
        model.add(tf.keras.layers.Dropout(DROPOUT)) # Reduce overfitting
        i += 1
    

    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def is_array_correct(array_content, indexes):
    total_length = 0
    if not indexes[0]:
        total_length = int(indexes[1])
    elif not indexes[1]:
        total_length = int(indexes[0])
    else:
        total_length = int(indexes[0]) * int(indexes[1])
    
    return total_length == len(array_content.split(","))


def save_raw_weights_to_file(model, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    with open(file_name, "a") as params_file:
        for index, layer in enumerate(model.layers):
            if len(layer.get_weights()) > 0:
                # print(f"layer {index}\n", layer.get_weights())
                for count, ele in enumerate(["weights", "biases"]):
                    params_file.write(f"\n\n\nlayer {index} - {ele}\n\n")
                    weights_content = np.transpose(layer.get_weights()[count])
                    params_file.write(str(weights_content.shape) + "\n\n")
                    np.savetxt(params_file, weights_content, fmt="%.9f", delimiter=", ")
                    # print(layer.get_weights()[count].shape)


def convert_to_c(file_name):
    converted_content = ""
    with open(file_name, 'r') as params_file:
        lines = params_file.readlines()
        curr_array_len = actual_array_len = 0
        array_content  = ""
        for line in lines:
            if not line or line.isspace() or line.startswith("layer"):
                converted_content += line
                continue
            if line.startswith("("):
                indexes = line.replace("(", "").replace(")", "").strip().split(",")
                actual_array_len = int(indexes[0].strip())
                curr_array_len = 0
                array_content = ""
                continue
            
            # if line.count(",") > 1:
            #     if curr_array_len == (actual_array_len - 1):
            #         array_content += ("{" + line.strip() + "}\n")
            #         converted_content += array_content
            #     else:
            #         array_content += ("{" + line.strip() + "},\n")
            # else:
            #     if curr_array_len == 0:
            #         array_content += ("{" + line.strip() + ", \n")
            #     elif curr_array_len == (actual_array_len - 1):
            #         array_content += (line.strip() + "}\n")
            #         assert is_array_correct(array_content, indexes)
            #         converted_content += array_content
            #     else:
            #         array_content += (line.strip() + ", ")

            if curr_array_len == 0:
                    array_content += ("{" + line.strip() + ", \n")
            elif curr_array_len == (actual_array_len - 1):
                array_content += (line.strip() + "}\n")
                assert is_array_correct(array_content, indexes)
                converted_content += array_content
            else:
                array_content += (line.strip() + ", ")

            curr_array_len += 1

    with open(file_name, "w") as params_file:
        params_file.write(converted_content)


def extract_params(model, file_name):
    save_raw_weights_to_file(model, file_name)
    convert_to_c(file_name)


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


def extract_features(input):
    features = []
    for i in range(NUM_DATA):
        axis = input[:, i]
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

    return np.array(features, dtype=np.int32)


def summarise_data(data_file, label_file, lines_per_matrix):
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

    # Extract column-wise summary features
    features = [extract_features(m) for m in matrices]

    return np.array(features), np.array(labels_one_hot)


def main():
    should_generate_data = input("Generate dummy data? Y/N: ")
    if should_generate_data.upper() == "Y":
        generate_dummy_data(f"{DATA_FOLDER_NAME}/data.txt", f"{DATA_FOLDER_NAME}/label.txt")
    
    
    X, Y = summarise_data(f"{DATA_FOLDER_NAME}/data.txt", f"{DATA_FOLDER_NAME}/label.txt", WINDOW_SIZE)
    print(f"X: size {X.shape}\n{X}")
    print(f"Y: size {Y.shape}\n{Y}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True, stratify=Y.argmax(axis=1))
    print("Training set:", X_train.shape, Y_train.shape)
    print("Test set:", X_test.shape, Y_test.shape)


    # Perform 3-Fold Cross Validation
    # kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # fold = 1
    # cv_scores = []

    # for train_idx, val_idx in kf.split(X_train):
    #     print(f"\n--- Fold {fold} ---")
        
    #     X_tr, X_val = X_train[train_idx], X_train[val_idx]
    #     Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]

    #     model = get_model()  # create a new instance each fold
    #     print(model.summary())

    #     # Train on training fold
    #     model.fit(X_tr, Y_tr, epochs=EPOCHS, verbose=0)

    #     # Validate on validation fold
    #     score = model.evaluate(X_val, Y_val, verbose=0)
    #     print(f"Validation score (fold {fold}):\n  Loss - {score[0]}, Accuracy - {score[1]}")
    #     cv_scores.append(score)

    #     fold += 1

    # cv_scores = np.array(cv_scores)
    # print(f"\nAverage CV score:\n  Loss - {np.mean(cv_scores[:, 0])}, Accuracy - {np.mean(cv_scores[:, 1])}")


    # Train on full training set, then evaluate on final test set
    final_model = get_model()
    print(final_model.summary())
    final_model.fit(X_train, Y_train, epochs=EPOCHS)
    test_score = final_model.evaluate(X_test, Y_test, verbose=2)
    print(f"Final test score:\n  Loss - {test_score[0]}, Accuracy - {test_score[1]}")

    # Create confusion matrix
    y_pred_probs = final_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", confusion_matrix)

    # Print classification report with all values
    print(metrics.classification_report(y_true, y_pred))

    # Heat map
    sns.heatmap(confusion_matrix, cmap="Blues", annot=True, 
                cbar_kws={"orientation":"vertical", "label": "Number of readings"},
                xticklabels=DATA_LABELS, yticklabels=DATA_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Export params and model
    should_export_model = input("Export params and model? Y/N: ")
    if should_export_model.upper() == "Y":
        extract_params(final_model, f"{EXPORT_FOLDER_NAME}/params.txt")
        final_model.save("{EXPORT_FOLDER_NAME}/my_model.keras")
        print("Data saved!")


if __name__ == "__main__":
    main()

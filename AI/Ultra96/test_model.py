from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np
import time
import logging

MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"


logger = logging.getLogger("ai_engine")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def setup_ai():
    global DATA_LABELS, NUM_DATA, NUM_SENSORS
    global inputBuffer, outputBuffer, dma, mode

    NUM_OF_PACKETS = 20
    NUM_DATA = 6

    mode = int(input("Enter a mode number: "))

    PL.reset() # Reset the programmable logic
    logger.info("Programmable Logic has been reset.")

    if mode == 1:
        ol = Overlay('design_1.bit') # Loads the FPGA bitstream
        logger.info("Overlay loaded (design_1.bit): %s", ol)
        DATA_LABELS = ["Idle", "Raise left arm", "Raise right arm", "Raise both arms", "Wave left hand", "Wave right hand", 
                       "Wave both hands", "Left arm circle", "Right arm circle", "Both arms circles", "Clap", "Star jump"]
        NUM_SENSORS = 2
    else:
        ol = Overlay('design_2.bit')
        logger.info("Overlay loaded (design_2.bit): %s", ol)
        DATA_LABELS = ["Idle", "Shake left hand", "Shake right hand", "Shake both hands", "Left high-five", "Right high-five", "Both high-five"]
        NUM_SENSORS = 4

    NUM_INPUT = 8 * NUM_DATA * 2 if MODEL_TYPE == "Simplified MLP" else NUM_OF_PACKETS * NUM_DATA * 2
    
    dma = ol.axi_dma_0 # Direct memory access channel between FPGA and ARM
    logger.info("DMA object: %s", dma)

    if MODEL_TYPE == "Simplified MLP":
        inputBuffer = allocate(shape=(NUM_INPUT,), dtype=np.float32)
    else:
        inputBuffer = allocate(shape=(NUM_INPUT,), dtype=np.int32) # To store input data to send to FPGA
    outputBuffer = allocate(shape=(len(DATA_LABELS),), dtype=np.float32) # To store output logit from FPGA

    logger.info("Input buffer allocated with shape %s", inputBuffer.shape)
    logger.info("Output buffer allocated with shape %s", outputBuffer.shape)


def extract_features(input):
    features = []
    for i in range(NUM_DATA):
        axis = input[:, i]
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


def get_model_output(inputArray):
    global inputBuffer, outputBuffer, dma, DATA_LABELS
    logger.info("Preparing input buffer...")
    np.copyto(inputBuffer, inputArray)

    try:
        logger.info("Starting DMA send transfer...")
        dma.sendchannel.transfer(inputBuffer)
        dma.recvchannel.transfer(outputBuffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        logger.info("DMA receive completed.")

        return outputBuffer.copy()
    except RuntimeError as e:
        print(e)
        print("Error config:\n", dma.register_map)


def main():
    buckets = [[] for _ in range(NUM_SENSORS)]
    goldenLogitsMatrix = np.loadtxt(f"golden_logits_{mode}.txt", dtype=np.float32) # Output from testing on laptop

    sampleCount = 0
    numFailures = 0
    numLogitMismatches = 0
    totalComputeTime = 0.0

    interactiveInput = input("Interactive mode? Y/N: ")
    interactiveMode = interactiveInput.upper() == "Y"

    def classify_action(localBuckets):
        nonlocal sampleCount, numFailures, numLogitMismatches, totalComputeTime
        logger.info("Received new input data")

        # Form inputArray
        if MODEL_TYPE == "Simplified MLP":
            inputArray = np.array([extract_features(np.array(bucket)) for bucket in localBuckets], dtype=np.float32).ravel()
        elif MODEL_TYPE == "MLP" or MODEL_TYPE == "RNN":
            inputArray = np.concatenate(localBuckets, axis=0).ravel().astype(np.int32)
        elif MODEL_TYPE == "CNN":
            inputArray = np.concatenate(localBuckets, axis=0).T.ravel().astype(np.int32)
        else:
            raise ValueError("Invalid MODEL_TYPE")
        
        startTime = time.time()
        predLogits = get_model_output(inputArray)
        predClass = int(np.argmax(predLogits))
        endTime = time.time()
        totalComputeTime += (endTime - startTime)

        logger.info("Prediction logits: %s", predLogits)
        logger.info("Predicted class: %d %s", predClass, DATA_LABELS[predClass])

        goldenLogits = goldenLogitsMatrix[sampleCount]
        goldenClass = int(np.argmax(goldenLogits))

        # Compare output from Ultra96 and laptop
        if predClass != goldenClass:
            numFailures += 1

        if np.any(np.abs(predLogits - goldenLogits) > 0.01):
            numLogitMismatches += 1

        sampleCount += 1

        if sampleCount % 50 == 0:
            print(f"Processed {sampleCount} samples so far...")

        if not interactiveMode:
            return True
        
        continueSignal = input("Continue? Y/N: ")
        return continueSignal.upper() == "Y"
    
    with open(f"data_{mode}.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of a matrix
                if any(buckets):
                    if NUM_SENSORS == 2:
                        if not classify_action(buckets):
                            break
                    else:
                        if not classify_action(buckets[:2]):
                            break
                        if not classify_action(buckets[2:]):
                            break
                    buckets = [[] for _ in range(NUM_SENSORS)]
                continue
            else:
                lineValues = [int(x) for x in line.split(" ")]
                deviceID = lineValues[0]
                sensorValues = lineValues[1:]
                buckets[deviceID - 1].append(sensorValues)
                
        if any(buckets): # Handle last matrix if file does not end with empty line
            if NUM_SENSORS == 2:
                classify_action(buckets)
            else:
                if classify_action(buckets[:2]):
                    classify_action(buckets[2:])

    # Print summary
    print(f"Processed {sampleCount} samples")
    
    if sampleCount > 0:
        print(f"Average time per prediction: {totalComputeTime/sampleCount:.6f} seconds")

    if numFailures == 0:
        print("Class check passed! All predicted classes match the golden.")
    else:
        print(f"Class check failed! {numFailures} mismatches found.")

    if numLogitMismatches == 0:
        print("Logit check passed! All logits within 0.01 tolerance.")
    else:
        print(f"Logit check failed! {numLogitMismatches} values exceeded 0.01 difference.")


if __name__ == "__main__":
    setup_ai()
    main()

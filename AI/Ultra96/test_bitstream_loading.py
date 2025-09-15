from pynq import Overlay

# Load your bitstream + hwh
ol = Overlay("design_1.bit")

# Check if overlay loaded successfully
print("Overlay loaded:", ol.is_loaded())

# Print available memory-mapped registers
print("Available MMIO regions:", list(ol.ip_dict.keys()))

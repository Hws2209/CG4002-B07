#include <Wire.h>

#define MPU_ADDR 0x68

#define USE_FILTER true   // set false for raw, true for EMA Filter

bool printData = true; // flag to control printing

const float G = 9.80665f;
const float ACC_LSB_PER_G[4]    = {16384.0, 8192.0, 4096.0, 2048.0};
const float GYRO_LSB_PER_DEG[4] = {131.0, 65.5, 32.8, 16.4};

const uint8_t ACC_FS = 1;   // ±4g
const uint8_t GYRO_FS = 1;  // ±500 dps

// --- Noise filter settings ---
#define ALPHA 0.3f   // smoothing factor (closer to 1 = faster response)
#define ACCEL_THRESHOLD 2.0f    // m/s^2 (tune!)
#define GYRO_THRESHOLD  10.0f   // deg/s (tune!)

float avgAccel = 0, avgGyro = 0;

struct RawMPU {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

// Helpers
float magnitude(float x, float y, float z) {
  return sqrt(x * x + y * y + z * z);
}

void writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

void readRegs(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, len, true);
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
}

void setupMPU() {
  writeReg(MPU_ADDR, 0x6B, 0x00);  // Wake up
  delay(10);
  writeReg(MPU_ADDR, 0x1C, ACC_FS << 3);  // Accel FS
  writeReg(MPU_ADDR, 0x1B, GYRO_FS << 3); // Gyro FS
  writeReg(MPU_ADDR, 0x1A, 0x03);         // DLPF
}

RawMPU readRawMPU() {
  uint8_t buf[6];
  RawMPU d;

  // accel
  readRegs(MPU_ADDR, 0x3B, buf, 6);
  d.ax = (int16_t)(buf[0] << 8 | buf[1]);
  d.ay = (int16_t)(buf[2] << 8 | buf[3]);
  d.az = (int16_t)(buf[4] << 8 | buf[5]);

  // gyro
  readRegs(MPU_ADDR, 0x43, buf, 6);
  d.gx = (int16_t)(buf[0] << 8 | buf[1]);
  d.gy = (int16_t)(buf[2] << 8 | buf[3]);
  d.gz = (int16_t)(buf[4] << 8 | buf[5]);

  return d;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  setupMPU();
  Serial.println("MPU6050 with raw output + threshold filter ready.");
}

void loop() {
  static unsigned long lastSample = 0;
  static unsigned long lastPrint  = 0;
  unsigned long now = millis();

  // --- Check Serial for toggle ---
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {  
      printData = !printData;      // toggle printing
      if(printData) Serial.println("Printing resumed.");
      else Serial.println("Printing stopped.");
    }
  }

  // --- Sample sensor at 50 Hz (every 20 ms) ---
  if (now - lastSample >= 20) {
    lastSample = now;

    RawMPU raw = readRawMPU();

    // Convert raw to physical units for filtering only
    float ax = raw.ax / ACC_LSB_PER_G[ACC_FS] * G;
    float ay = raw.ay / ACC_LSB_PER_G[ACC_FS] * G;
    float az = raw.az / ACC_LSB_PER_G[ACC_FS] * G;
    float gx = raw.gx / GYRO_LSB_PER_DEG[GYRO_FS];
    float gy = raw.gy / GYRO_LSB_PER_DEG[GYRO_FS];
    float gz = raw.gz / GYRO_LSB_PER_DEG[GYRO_FS];

    float amag = magnitude(ax, ay, az);
    float gmag = magnitude(gx, gy, gz);

    // Exponential moving average
    avgAccel = ALPHA * amag + (1.0f - ALPHA) * avgAccel;
    avgGyro  = ALPHA * gmag  + (1.0f - ALPHA) * avgGyro;

    bool significant = (fabs(amag - avgAccel) > ACCEL_THRESHOLD ||
                        fabs(gmag - avgGyro) > GYRO_THRESHOLD);

    // --- Print only at 10 Hz (every 100 ms) ---
    if (printData && (now - lastPrint >= 100)) {
      lastPrint = now;

      if (significant) {
        // Print raw integers
        Serial.print(raw.ax); Serial.print(" ");
        Serial.print(raw.ay); Serial.print(" ");
        Serial.print(raw.az); Serial.print(" ");
        Serial.print(raw.gx); Serial.print(" ");
        Serial.print(raw.gy); Serial.print(" ");
        Serial.println(raw.gz);
      } else {
        // Print six zeros
        Serial.println("0 0 0 0 0 0");
      }
    }
  }
}

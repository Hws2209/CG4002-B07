#include <Wire.h>

#define MPU_ADDR 0x68

const float G = 9.80665f;
const float ACC_LSB_PER_G[4]    = {16384.0, 8192.0, 4096.0, 2048.0};
const float GYRO_LSB_PER_DEG[4] = {131.0, 65.5, 32.8, 16.4};

const uint8_t ACC_FS = 1;   // ±4g
const uint8_t GYRO_FS = 1;  // ±500 dps

// --- Noise filter settings ---
#define ALPHA 0.3f   // smoothing factor (closer to 1 = faster response)
#define WINDOW_SIZE     10
#define ACCEL_THRESHOLD 2.0f    // m/s^2 (tune!)
#define GYRO_THRESHOLD  10.0f    // deg/s (tune!)
#define ACTION_LENGTH   5


float avgAccel = 0, avgGyro = 0;

// FIFO + magnitudes
float accelMagnitude[WINDOW_SIZE];
float gyroMagnitude[WINDOW_SIZE];
int buffer_index = 0;

struct MPUdata {
  float ax, ay, az;
  float gx, gy, gz;
};

// Helpers
float magnitude(float x, float y, float z) {
  return sqrt(x*x + y*y + z*z);
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
  for(uint8_t i=0;i<len;i++) buf[i] = Wire.read();
}

void setupMPU() {
  writeReg(MPU_ADDR, 0x6B, 0x00);
  delay(10);
  writeReg(MPU_ADDR, 0x1C, ACC_FS << 3);
  writeReg(MPU_ADDR, 0x1B, GYRO_FS << 3);
  writeReg(MPU_ADDR, 0x1A, 0x03);
}

MPUdata readMPU() {
  uint8_t buf[6];
  MPUdata d;

  // accel
  readRegs(MPU_ADDR, 0x3B, buf, 6);
  int16_t ax_raw = (int16_t)(buf[0]<<8 | buf[1]);
  int16_t ay_raw = (int16_t)(buf[2]<<8 | buf[3]);
  int16_t az_raw = (int16_t)(buf[4]<<8 | buf[5]);

  d.ax = ax_raw / ACC_LSB_PER_G[ACC_FS] * G;
  d.ay = ay_raw / ACC_LSB_PER_G[ACC_FS] * G;
  d.az = az_raw / ACC_LSB_PER_G[ACC_FS] * G;

  // gyro
  readRegs(MPU_ADDR, 0x43, buf, 6);
  int16_t gx_raw = (int16_t)(buf[0]<<8 | buf[1]);
  int16_t gy_raw = (int16_t)(buf[2]<<8 | buf[3]);
  int16_t gz_raw = (int16_t)(buf[4]<<8 | buf[5]);

  d.gx = gx_raw / GYRO_LSB_PER_DEG[GYRO_FS];
  d.gy = gy_raw / GYRO_LSB_PER_DEG[GYRO_FS];
  d.gz = gz_raw / GYRO_LSB_PER_DEG[GYRO_FS];

  return d;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  setupMPU();

  for(int i=0;i<WINDOW_SIZE;i++) {
    accelMagnitude[i] = 0;
    gyroMagnitude[i] = 0;
  }

  Serial.println("MPU6050 with threshold filter ready.");
}




void loop() {
  MPUdata data = readMPU();

  // Magnitudes
  float amag = magnitude(data.ax, data.ay, data.az);
  float gmag = magnitude(data.gx, data.gy, data.gz);

  // Exponential moving average (reacts quickly to changes)
  avgAccel = ALPHA * amag + (1.0f - ALPHA) * avgAccel;
  avgGyro  = ALPHA * gmag  + (1.0f - ALPHA) * avgGyro;

  // Threshold check (raw vs smoothed)
  if (fabs(amag - avgAccel) > ACCEL_THRESHOLD ||
      fabs(gmag - avgGyro) > GYRO_THRESHOLD) {

    Serial.print("Acc [m/s^2]: ");
    Serial.print(data.ax,2); Serial.print(", ");
    Serial.print(data.ay,2); Serial.print(", ");
    Serial.println(data.az,2);

    Serial.print("Gyro [deg/s]: ");
    Serial.print(data.gx,2); Serial.print(", ");
    Serial.print(data.gy,2); Serial.print(", ");
    Serial.println(data.gz,2);
    Serial.println("---");
  }

  delay(100); // ~10Hz sample rate
}

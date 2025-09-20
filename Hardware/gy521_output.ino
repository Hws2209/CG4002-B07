#include <Wire.h>

#define MPU_ADDR 0x68    // MPU6050 I2C address (AD0 pin LOW)

const float G = 9.80665f; // gravity in m/s^2
const float DEG2RAD = 3.14159265359f / 180.0f;

// Accel and Gyro LSB per full scale
const float ACC_LSB_PER_G[4]    = {16384.0, 8192.0, 4096.0, 2048.0}; // ±2, ±4, ±8, ±16g
const float GYRO_LSB_PER_DEG[4] = {131.0, 65.5, 32.8, 16.4};         // ±250, ±500, ±1000, ±2000 dps

// Choose your full-scale ranges
const uint8_t ACC_FS = 3;   // ±4g
const uint8_t GYRO_FS = 3;  // ±500 deg/s

// Raw sensor data
int16_t ax_raw, ay_raw, az_raw;
int16_t gx_raw, gy_raw, gz_raw;

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
  // Wake up MPU6050
  writeReg(MPU_ADDR, 0x6B, 0x00);
  delay(10);

  // Set accel full-scale
  writeReg(MPU_ADDR, 0x1C, ACC_FS << 3);
  // Set gyro full-scale
  writeReg(MPU_ADDR, 0x1B, GYRO_FS << 3);
  // Optional: DLPF config (set to 3 for ~44Hz bandwidth)
  writeReg(MPU_ADDR, 0x1A, 0x03);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();  // SDA/SCL defaults for Arduino; use (21,22) on ESP32 if needed
  delay(50);

  setupMPU();

  Serial.println("MPU6050 (GY-521) ready.");
}

bool checkAccelClipping(int16_t ax, int16_t ay, int16_t az) {
  int16_t maxVal = ACC_LSB_PER_G[ACC_FS] * ((1 << ACC_FS) ? (1<<ACC_FS) : 1); // approx
  return (abs(ax) >= maxVal || abs(ay) >= maxVal || abs(az) >= maxVal);
}

void loop() {
uint8_t buf[6];

// --- Read accel ---
readRegs(MPU_ADDR, 0x3B, buf, 6);
ax_raw = (int16_t)(buf[0]<<8 | buf[1]);
ay_raw = (int16_t)(buf[2]<<8 | buf[3]);
az_raw = (int16_t)(buf[4]<<8 | buf[5]);

// --- Read gyro ---
readRegs(MPU_ADDR, 0x43, buf, 6);
gx_raw = (int16_t)(buf[0]<<8 | buf[1]);
gy_raw = (int16_t)(buf[2]<<8 | buf[3]);
gz_raw = (int16_t)(buf[4]<<8 | buf[5]);

// Convert to human-readable units
float ax = ax_raw / ACC_LSB_PER_G[ACC_FS] * G;
float ay = ay_raw / ACC_LSB_PER_G[ACC_FS] * G;
float az = az_raw / ACC_LSB_PER_G[ACC_FS] * G;

float gx = gx_raw / GYRO_LSB_PER_DEG[GYRO_FS];
float gy = gy_raw / GYRO_LSB_PER_DEG[GYRO_FS];
float gz = gz_raw / GYRO_LSB_PER_DEG[GYRO_FS];

// --- Check for clipping ---
bool accelClipping = (abs(ax_raw) == 32767 || abs(ay_raw) == 32767 || abs(az_raw) == 32767);
bool gyroClipping  = (abs(gx_raw) == 32767 || abs(gy_raw) == 32767 || abs(gz_raw) == 32767);

if(accelClipping) Serial.println("!!! Accel clipping detected !!!");
if(gyroClipping)  Serial.println("!!! Gyro clipping detected !!!");

// Print values
Serial.print("Acc [m/s^2]: "); Serial.print(ax,3); Serial.print(", "); Serial.print(ay,3); Serial.print(", "); Serial.println(az,3);
Serial.print("Gyro [deg/s]: "); Serial.print(gx,2); Serial.print(", "); Serial.print(gy,2); Serial.print(", "); Serial.println(gz,2);

delay(100);
}

/**
  MPUdata data = readMPU();
  putToFIFO(&data); 
  float avgAccelMagnitude = 0.0; 
  float avgGyroMagnitude 0.0; 
  for (int i = 0; i < WINDOW _SIZE; i++) {
    avgAccelMagnitude += accelMagnitude[i]; 
    avgGyroMagnitude += gyroMagnitude[i];
    }
  avgAccelMagnitude /= WINDOW_SIZE; 
  avgGyroMagnitude /= WINDOW_SIZE; 
  float amag = MAGNITUDE(data.ax, data.ay, data.az); 
  float gmag - MAGNITUDE(data.gx, data.gy, data.gz);
  
  accelMagnitude[buffer_index] = amag;
  gyroMagnitude[buffer_index] = gmag; 
  buffer_index = (buffer_index + 1) % WINDOW_SIZE;

  if (abs(avgAccelMagnitude - amag) > ACCEL_THRESHOLD ||
      abs (avgGyroMagnitude - gmag) > GYRO_THRESHOLD) {
    for (uint16_t i = 0; i < ACTION_LENGTH; i++) {
      MPUdata dataToSend = readDataFromFIFO();
      sendCurrMPU(dataToSend);
      delay(50); 
      MPUdata newData = readMPU();
      putToFIFO(&newData);
      accelMagnitude[buffer_index] = MAGNITUDE(newData.ax, newData.ay, newData.az);
      gyroMagnitude[buffer_index] =MAGNITUDE(newData.gx, newData.gy, newData.gz);
      buffer_index = (buffer_index + 1) % WINDOW_SIZE;
      // wdt_reset();
    }
    delay(200);
*/


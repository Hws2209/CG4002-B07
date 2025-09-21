#include <Wire.h>

#define MPU_ADDR 0x68

bool printData = true; // flag to control printing
uint8_t buf[6];

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
  writeReg(MPU_ADDR, 0x6B, 0x00);  // wake up
  delay(10);
  writeReg(MPU_ADDR, 0x1C, 1 << 3); // accel ±4g
  writeReg(MPU_ADDR, 0x1B, 1 << 3); // gyro ±500 dps
  writeReg(MPU_ADDR, 0x1A, 0x03);   // DLPF ~44Hz
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  setupMPU();
  Serial.println("ax ay az gx gy gz");
  Serial.println("Send 's' to stop/start printing");
}

void loop() {
  // --- Check Serial for toggle ---
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {  
      printData = !printData;      // toggle printing
      if(printData) Serial.println("Printing resumed.");
      else Serial.println("Printing stopped.");
    }
  }

  int16_t ax, ay, az, gx, gy, gz;

  // --- Read accel ---
  readRegs(MPU_ADDR, 0x3B, buf, 6);
  ax = (int16_t)(buf[0]<<8 | buf[1]);
  ay = (int16_t)(buf[2]<<8 | buf[3]);
  az = (int16_t)(buf[4]<<8 | buf[5]);

  // --- Read gyro ---
  readRegs(MPU_ADDR, 0x43, buf, 6);
  gx = (int16_t)(buf[0]<<8 | buf[1]);
  gy = (int16_t)(buf[2]<<8 | buf[3]);
  gz = (int16_t)(buf[4]<<8 | buf[5]);

  // --- Print raw data if enabled ---
  if (printData) {
    Serial.print(ax); Serial.print(" ");
    Serial.print(ay); Serial.print(" ");
    Serial.print(az); Serial.print(" ");
    Serial.print(gx); Serial.print(" ");
    Serial.print(gy); Serial.print(" ");
    Serial.println(gz);
  }

  delay(100); // ~10Hz
}

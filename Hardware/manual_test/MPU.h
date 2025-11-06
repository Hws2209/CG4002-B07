#ifndef MPU_H
#define MPU_H

#include <Wire.h>
#include <Arduino.h>

#define MPU_ADDR 0x68

// --- MPU configuration ---
const float G = 9.80665f;
const float ACC_LSB_PER_G[4]    = {16384.0, 8192.0, 4096.0, 2048.0};
const float GYRO_LSB_PER_DEG[4] = {131.0, 65.5, 32.8, 16.4};

const uint8_t ACC_FS = 1;   // ±4g
const uint8_t GYRO_FS = 1;  // ±500 dps

#define USE_FILTER true
#define ALPHA 0.3f
#define ACCEL_THRESHOLD 2.0f
#define GYRO_THRESHOLD 10.0f

// --- Sensor packet ---
struct SensorPacket {
  uint16_t header;
  uint16_t device_id;
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  int32_t padding;
};

class MPU {
  public:
    float avgAccel = 0;
    float avgGyro  = 0;
    bool useFilter = true; //true;

    void begin() {
      Wire.begin();
      // Wake MPU
      writeReg(0x6B, 0x00);
      delay(10);
      writeReg(0x1C, ACC_FS << 3);
      writeReg(0x1B, GYRO_FS << 3);
      writeReg(0x1A, 0x03);
    }

    SensorPacket readFilteredPacket(uint16_t device_id) {
      RawMPU raw = readRaw();
      
      if (!useFilter) {
        return toPacket(raw, device_id);
      }

      // Convert raw to physical units
      float ax = raw.ax / ACC_LSB_PER_G[ACC_FS] * G;
      float ay = raw.ay / ACC_LSB_PER_G[ACC_FS] * G;
      float az = raw.az / ACC_LSB_PER_G[ACC_FS] * G;
      float gx = raw.gx / GYRO_LSB_PER_DEG[GYRO_FS];
      float gy = raw.gy / GYRO_LSB_PER_DEG[GYRO_FS];
      float gz = raw.gz / GYRO_LSB_PER_DEG[GYRO_FS];

      float amag = magnitude(ax, ay, az);
      float gmag = magnitude(gx, gy, gz);

      // EMA
      avgAccel = ALPHA * amag + (1.0f - ALPHA) * avgAccel;
      avgGyro  = ALPHA * gmag + (1.0f - ALPHA) * avgGyro;

      bool significant = (fabs(amag - avgAccel) > ACCEL_THRESHOLD ||
                          fabs(gmag - avgGyro) > GYRO_THRESHOLD);

      SensorPacket pkt;
      pkt.header = 0xAA55;
      pkt.device_id = device_id;

      if (significant) {
        pkt.ax = raw.ax;
        pkt.ay = raw.ay;
        pkt.az = raw.az;
        pkt.gx = raw.gx;
        pkt.gy = raw.gy;
        pkt.gz = raw.gz;
      } else {
        pkt.ax = pkt.ay = pkt.az = pkt.gx = pkt.gy = pkt.gz = 0;
      }

      return pkt;
    }

  private:
    struct RawMPU { int16_t ax, ay, az; int16_t gx, gy, gz; };

    float magnitude(float x, float y, float z) { return sqrt(x*x + y*y + z*z); }

    void writeReg(uint8_t reg, uint8_t val) {
      Wire.beginTransmission(MPU_ADDR);
      Wire.write(reg);
      Wire.write(val);
      Wire.endTransmission(true);
    }

    void readRegs(uint8_t reg, uint8_t *buf, uint8_t len) {
      Wire.beginTransmission(MPU_ADDR);
      Wire.write(reg);
      Wire.endTransmission(false);
      Wire.requestFrom(MPU_ADDR, len, true);
      for (uint8_t i=0;i<len;i++) buf[i] = Wire.read();
    }

    RawMPU readRaw() {
      uint8_t buf[6];
      RawMPU d;
      readRegs(0x3B, buf, 6); d.ax = (buf[0]<<8)|buf[1]; d.ay = (buf[2]<<8)|buf[3]; d.az = (buf[4]<<8)|buf[5];
      readRegs(0x43, buf, 6); d.gx = (buf[0]<<8)|buf[1]; d.gy = (buf[2]<<8)|buf[3]; d.gz = (buf[4]<<8)|buf[5];
      return d;
    }

    SensorPacket toPacket(RawMPU raw, uint16_t device_id) {
      SensorPacket pkt;
      pkt.header = 0xAA55;
      pkt.device_id = device_id;
      pkt.ax = raw.ax; pkt.ay = raw.ay; pkt.az = raw.az;
      pkt.gx = raw.gx; pkt.gy = raw.gy; pkt.gz = raw.gz;
      return pkt;
    }
};

#endif

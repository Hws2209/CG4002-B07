#include <Arduino.h>
#include "Display.h"
#include "MPU.h"

// Global objects
Display display;
MPU mpu;
SensorPacket packet;

int mode = 0;
int actualClass = -1;
unsigned long lastDisplay = 0;
unsigned long lastSample = 0;

void setup() {
  Serial.begin(115200);
  while(!Serial);

  Serial.println("=== Gesture AI Test Mode ===");
  Serial.println("Commands:");
  Serial.println("  mode <num>   -> set test mode");
  Serial.println("  action <num> -> simulate action class received");
  Serial.println("  print        -> show current settings");
  Serial.println();

  // Simulate Display initialization
  display.beginDevice2();  // or display.beginDevice3();
  
  // Simulate MPU
  mpu.begin();

  // Initialize packet
  packet.header = 0xAA55;
  packet.device_id = 4;

  Serial.println("Ready. Type command below:");
}

void loop() {
  unsigned long now = millis();

  if (Serial.available()) {
      String input = Serial.readStringUntil('\n');
      input.trim();

      if (input.startsWith("mode")) {
          mode = input.substring(5).toInt();
          Serial.printf("Mode set to %d\n", mode);
          display.showActionClass(actualClass, mode); // redraw
      } 
      else if (input.startsWith("action")) {
          actualClass = input.substring(7).toInt();
          Serial.printf("Action class set to %d\n", actualClass);
          display.showActionClass(actualClass, mode); // redraw
      }
  }


  // 2️⃣ Simulate IMU readings at 50 Hz
  if (now - lastSample >= 20) {
    lastSample = now;
    packet = mpu.readFilteredPacket(packet.device_id);
  }

  // 3️⃣ Simulate display updates at 10 Hz
  if (now - lastDisplay >= 100) {
    lastDisplay = now;


  }
}

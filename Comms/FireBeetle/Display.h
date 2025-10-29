#ifndef DISPLAY_H
#define DISPLAY_H

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DFRobot_MAX17043.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDR 0x3C

class Display {
  public:
    Adafruit_SSD1306 oled = Adafruit_SSD1306(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
    DFRobot_MAX17043 gauge;

    unsigned long lastOledUpdate = 0;
    float batteryPercent = 0;
    float batteryVoltage = 0;

    void begin() {
      // OLED init
      if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        Serial.println("⚠️ OLED init failed!");
      } else {
        oled.clearDisplay();
        oled.setTextSize(1);
        oled.setTextColor(SSD1306_WHITE);
        oled.setCursor(0, 0);
        oled.println("OLED Ready");
        oled.display();
      }

      // Battery gauge init
      gauge.begin();

      // Manual QuickStart for MAX17043
      Wire.beginTransmission(0x36);
      Wire.write(0x06);
      Wire.write(0x40);
      Wire.write(0x00);
      Wire.endTransmission();

      delay(100);
      updateBattery();
      updateDisplay(); // show first reading right away
    }

    void updateBattery() {
      batteryPercent = gauge.readPercentage();
      batteryVoltage = gauge.readVoltage();
    }

    void updateDisplay() {
      oled.clearDisplay();
      oled.setTextSize(2);
      oled.setTextColor(SSD1306_WHITE);
      oled.setCursor(0, 20);

      oled.printf("%.0f%%", batteryPercent);
      oled.display();
    }

    void loopUpdate() {
      unsigned long now = millis();

      // Update OLED and battery every 1s
      if (now - lastOledUpdate >= 1000) {
        lastOledUpdate = now;
        updateBattery();
        updateDisplay();  
      }
    }
};

#endif

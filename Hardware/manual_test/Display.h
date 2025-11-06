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

    unsigned long lastUpdate = 0;
    float batteryPercent = 0;
    float batteryVoltage = 0;
    String playerLabel = "";

    // ===== Common Init =====
    void beginOLED() {
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
    }

    void showActionClass(int classIndex, int mode) {
        static const char* mode1Labels[] = {
            "Idle", "Wave left hand", "Wave right hand", "Wave both hands",
            "Left back arm circle", "Right back arm circle", "Both back arm circles",
            "Left front arm circle", "Right front arm \ncircle", "Both front arm circles",
            "Clap", "Star jump"
        };

        static const char* mode2Labels[] = {
            "Idle", "Shake left hand", "Shake right hand", "Shake both hands",
            "Left high-five", "Right high-five", "Both high-five"
        };

        const char** labels;
        int numLabels;

        if (mode == 1) {
            labels = mode1Labels;
            numLabels = sizeof(mode1Labels) / sizeof(mode1Labels[0]);
        } else {
            labels = mode2Labels;
            numLabels = sizeof(mode2Labels) / sizeof(mode2Labels[0]);
        }

        if (classIndex < 0 || classIndex >= numLabels) classIndex = 0;

        oled.clearDisplay();

        // Top row: player + battery
        oled.setTextSize(1);
        oled.setCursor(0, 0);
        oled.print(playerLabel);

        if (playerLabel == "Player 1") {
            updateBattery();
            oled.setCursor(90, 0);
            oled.printf("%.0f%%", batteryPercent);
        }

        // Action label
        oled.setTextSize(1);
        oled.setCursor(0, 20);
        oled.println("Action:");

        oled.setTextSize(1);
        oled.setCursor(0, 36);
        oled.println(labels[classIndex]);

        oled.display(); // <-- Important! flush buffer to OLED
    }



    // ===== Device 2 (Player 1, with battery gauge) =====
    void beginDevice2() {
    playerLabel = "Player 1";
    beginOLED();

    gauge.begin();

    // Manual QuickStart (reinitialize fuel gauge)
    Wire.beginTransmission(0x36);
    Wire.write(0x06);
    Wire.write(0x40);
    Wire.write(0x00);
    Wire.endTransmission();

    // Add a slightly longer delay and discard the first reading
    delay(500);
    gauge.readPercentage(); // dummy read to stabilize
    delay(100);

    updateBattery();
    showBatteryScreen();
    }

    void updateBattery() {
      batteryPercent = gauge.readPercentage();
      batteryVoltage = gauge.readVoltage();
    }

    void showBatteryScreen() {
      oled.clearDisplay();
      oled.setTextSize(1);
      oled.setTextColor(SSD1306_WHITE);
      oled.setCursor(0, 0);
      oled.println(playerLabel);

      oled.setTextSize(2);
      oled.setCursor(0, 24);
      oled.printf("%.0f%%", batteryPercent);
      oled.display();
    }

    void loopDevice2() {
      unsigned long now = millis();
      if (now - lastUpdate >= 1000) {
        lastUpdate = now;
        updateBattery();
        showBatteryScreen();
      }
    }

    // ===== Device 3 (Player 2, no gauge) =====
    void beginDevice3() {
      playerLabel = "Player 2";
      beginOLED();

      oled.clearDisplay();
      oled.setTextSize(1);
      oled.setTextColor(SSD1306_WHITE);
      oled.setCursor(0, 0);
      oled.println(playerLabel);
      oled.display();
    }

    void loopDevice3(const String &msg = "") {
      unsigned long now = millis();
      if (now - lastUpdate >= 1000) {
        lastUpdate = now;
        oled.clearDisplay();
        oled.setTextSize(1);
        oled.setTextColor(SSD1306_WHITE);
        oled.setCursor(0, 0);
        oled.println(playerLabel);
        oled.setCursor(0, 16);
        oled.println(msg);
        oled.display();
      }
    }
};

#endif

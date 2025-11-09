#include <CryptoAES_CBC.h>
#include <AES.h>
#include "MPU.h"
#include "Display.h"
#include <Arduino.h>
#include <WiFi.h>
#define SEND_DURATION 2000
#define DEVICE_ID 2
//#include <ESPping.h>     // Install "ESPping" library

//key[16] cotain 16 byte key(128 bit) for encryption
byte key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
//plaintext[16] contain the text we need to encrypt
byte plaintext[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
//cypher[16] stores the encrypted text
byte cypher[16];
//decryptedtext[16] stores decrypted text after decryption
byte decryptedtext[16];
//creating an object of AES128 class
AES128 aes128;



//WiFi setup
const char* ssid = "Wenwuuu";
const char* password = "11223344";
//const char* ssid = "Hws";
//const char* password = "22092003";
const char* host = "10.82.212.64";
const int port = 2105;
int packetCount = 0;
int mode;
int actualClass = 1;
int padding;


WiFiClient client;

SensorPacket packet;
MPU mpu;
Display display;

// put function declarations here:

void setup() {
  pinMode(D9, OUTPUT);
  digitalWrite(D9, HIGH);   // turn the LED on (HIGH is the voltage level)
  Serial.begin(115200);

  mpu.begin();
  if (DEVICE_ID == 2) display.beginDevice2();
  else if (DEVICE_ID == 3) display.beginDevice3();

  Serial.print("Testingg");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("WiFi Connected");
  digitalWrite(D9, LOW);

  //temporarily using mac address to differentiate firebeetles. may not be reliable.
  //on second thought, hardcode is way btr, easier to ensure consistency
  //  uint8_t mac[6];
  //  WiFi.macAddress(mac);
  //  packet.device_id = (mac[5] % 4)+1;
  //  Serial.print("Device ID: "); Serial.println(packet.device_id);
  Serial.print("Arduino IP: ");
  Serial.println(WiFi.localIP());

  // Step 1: Ping Windows host
  Serial.print("server IP:   ");
  Serial.println(host);
  Serial.print("Port to connect:   ");
  Serial.println(port);

  //  if (Ping.ping(host)) {
  //    Serial.println("Ping successful! Host reachable.");
  //  } else {
  //    Serial.println("Ping failed! Check network/firewall.");
  //  }

  //Connect to laptop
  if (!client.connect(host, port)) {
    Serial.println("Connection Failed");
  } else {
    Serial.println("Connected to Laptop");
    digitalWrite(D9, HIGH);
  }



  //handshake
  client.print("HELLO");
  while (client.available() == 0) {
    delay(10); //wait for server to reply
  }
  Serial.print("timeout setting:  ");
  Serial.println(client.getTimeout());
  //  String ack = client.readStringUntil('\n');
  String ack = client.readStringUntil('K');
  if (ack != "AC") {
    Serial.println("NOT Acknowledged");
    Serial.println("Reply from server: " + ack);
  }
  Serial.println("Acknowledged");
  while (client.available() == 0) {
    delay(10); //wait for server to reply
  }
  mode = client.read();
  Serial.print("Mode:   ");
  Serial.println(mode);


  client.write(DEVICE_ID);

  //test only, hardcode sensor value
  packet.header = 0xAA55;
  //packet.ax = 0xAB01;
  //packet.ay = 0xCD10;
  //packet.az = 0xEF11;
  //packet.gx = 0xAB02;
  //packet.gy = 0xCD20;
  //packet.gz = 0xEF22;
  packet.device_id = DEVICE_ID;
  packet.padding = 0;
  aes128.setKey(key, 16); // Setting Key for AES



}

void loop() {
  unsigned long lastSample = 0; // for 50 Hz sampling
  unsigned long lastSend = 0; // for 10 Hz sending
  unsigned long lastDisplay = 0; // 10Hz display update

  unsigned long now = millis();

  while (client.available() == 0) {
    now = millis();
    // Display update at 10 Hz (independent of packet send)
    if (now - lastDisplay >= 100) {
      lastDisplay = now;

      if (DEVICE_ID == 2) {
        display.loopDevice2();
      } else if (DEVICE_ID == 3) {
        display.loopDevice3("Hello!");
      }

      static int lastClass = -1;
      if ((DEVICE_ID == 2 || DEVICE_ID == 3) && actualClass != lastClass) {
        display.showActionClass(actualClass, mode);
        lastClass = actualClass;
      }
    }
    delay(10); //wait for server to reply
  }


  //  String reply = client.readStringUntil('\n');
  char reply = client.read();
  //  while (reply != "a") {
  //    reply = client.readStringUntil('\n');
  //    }
  //  if (reply != "a") { //beep test
  //    continue;
  //  }
  Serial.println("Reply from server: " + reply);
  unsigned long startTime = millis();
  packetCount = 0;
  padding = 0;
  //  packet.gz = 0xEF00;


  while (packetCount < 20) {
    now = millis();
    if (now - lastSample >= 20) {
      lastSample = now;
      // update packet with filtered MPU data
      packet = mpu.readFilteredPacket(DEVICE_ID);
    }

    // --- Send at 10 Hz (every 100 ms) ---
    if (now - lastSend >= 100) {
      lastSend = now;
      packet.padding = padding;
      padding += 1;
      aes128.encryptBlock(cypher, (byte*) & (packet.ax)); //cypher->output block and packet->input block
      memcpy((byte*) & (packet.ax), cypher, 16);
      client.write((uint8_t*)&packet, sizeof(packet)); // send most recent packet
      packetCount += 1;
    }
  }
  if (DEVICE_ID == 2 || DEVICE_ID == 3) {
    while (client.available() == 0) {
      delay(10); //wait for server to reply
    }
    actualClass = client.read();
    Serial.print("Actual Class: ");
    Serial.println(actualClass);
  }



  //  Serial.print("Number of packets sent: ");
  //  Serial.println(packetCount);

}

// put function definitions here:

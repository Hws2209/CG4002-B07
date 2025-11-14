#include <CryptoAES_CBC.h>
#include <AES.h>
#include "MPU.h"
#include "Display.h"
#include <Arduino.h>
#include <WiFi.h>
#define SEND_DURATION 2000
#define DEVICE_ID 4


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

uint8_t CHARKEY = 0x5A;

uint8_t xor_crypt(uint8_t encrypted_byte, uint8_t key) {
  return encrypted_byte ^ key;
}
//WiFi setup
//const char* ssid = "Wenwuuu";
//const char* password = "11223344";
const char* ssid = "Hws";
const char* password = "22092003";
const char* host = "10.224.54.64";
const int port = 2105;
int packetCount = 0;
int mode;
int actualClass = 0;
int padding;
int lastClass = -1;

WiFiClient client;

SensorPacket packet;
MPU mpu;
Display display;


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

  Serial.print("Arduino IP: ");
  Serial.println(WiFi.localIP());

  // Step 1: Ping Windows host
  Serial.print("server IP:   ");
  Serial.println(host);
  Serial.print("Port to connect:   ");
  Serial.println(port);

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
  mode = xor_crypt(mode, CHARKEY);
  Serial.print("Mode:   ");
  Serial.println(mode);


  client.write(xor_crypt(DEVICE_ID,CHARKEY));

  packet.header = 0xAA55;
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
#if (DEVICE_ID == 2) || (DEVICE_ID == 3)
    if (now - lastDisplay >= 100) {
      lastDisplay = now;
      display.showActionClass(actualClass, mode);
    }
#endif
    delay(10); //wait for server to reply
  }


  char reply = client.read();
  Serial.print("Reply from server: ");
  Serial.println(reply);
  if (reply == 'a') { //msg to start sending packets
    unsigned long startTime = millis();
    packetCount = 0;
    padding = 0;

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
      actualClass = xor_crypt(actualClass, CHARKEY);
      Serial.print("Actual Class: ");
      Serial.println(actualClass);
    }
  }
}

// put function definitions here:

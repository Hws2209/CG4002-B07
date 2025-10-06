#include <CryptoAES_CBC.h>
#include <AES.h>
#include "MPU.h"
#include <Arduino.h>
#include <WiFi.h>
#define SEND_DURATION 2000 
#define DEVICE_ID 2
//#include <ESPping.h>     // Install "ESPping" library

//key[16] cotain 16 byte key(128 bit) for encryption
byte key[16]={0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
//plaintext[16] contain the text we need to encrypt 
byte plaintext[16]={0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
//cypher[16] stores the encrypted text
byte cypher[16];
//decryptedtext[16] stores decrypted text after decryption
byte decryptedtext[16];
//creating an object of AES128 class
AES128 aes128;



//WiFi setup
const char* ssid = "Wenwuuu";
const char* password = "11223344";
const char* host = "192.168.100.64";
const int port = 2105;  
int packetCount = 0;

WiFiClient client;


//struct SensorPacket {
//  int16_t header;
//  int16_t deviceId; //denote left/right arm/leg, use 2 bytes for alignment
//  int16_t ax, ay, az;
//  int16_t gx, gy, gz;
//};

SensorPacket packet;
MPU mpu;

// put function declarations here:

void setup() {
  Serial.begin(115200);
  Serial.print("Testingg");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("WiFi Connected");

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
  if (!client.connect(host,port)){
    Serial.println("Connection Failed");
  } else {
    Serial.println("Connected to Laptop");  
  }
  


  //handshake
  client.print("HELLO");
  while(client.available()==0){
    delay(10); //wait for server to reply
  }
  String ack = client.readStringUntil('\n');
  if (ack != "ACK") {
    Serial.println("NOT Acknowledged");
  }
  Serial.println("Acknowledged");
  

  //test only, hardcode sensor value
  packet.header = 0xAA55;
//  packet.ax = 0xAB01;
//  packet.ay = 0xCD10;
//  packet.az = 0xEF11;
//  packet.gx = 0xAB02;
//  packet.gy = 0xCD20;
//  packet.gz = 0xEF22;
  packet.device_id= DEVICE_ID;
  aes128.setKey(key,16);// Setting Key for AES


}

void loop() {
  while(client.available()==0){
    delay(10); //wait for server to reply
  }

  String reply = client.readStringUntil('\n');
  Serial.println("Reply from server: " + reply);
  unsigned long startTime = millis();
  packetCount=0;
//  packet.gz = 0xEF00;
  unsigned long lastSample = 0; // for 50 Hz sampling
  unsigned long lastSend = 0; // for 10 Hz sending

  while (packetCount <20){
//    aes128.encryptBlock(cypher,(byte*)&packet);//cypher->output block and packet->input block
//    client.write((uint8_t*)cypher, sizeof(packet)); // 16 bytes
//    packet.gz += 1;
    unsigned long now = millis();
    if (now - lastSample >= 20) {
      lastSample = now;
      // update packet with filtered MPU data
      packet = mpu.readFilteredPacket(1);
    }
    
      // --- Send at 10 Hz (every 100 ms) ---
    if (now - lastSend >= 100) {
      lastSend = now;
      client.write((uint8_t*)&packet, sizeof(packet)); // send most recent packet
      packetCount+=1;
    }
  }
  Serial.print("Number of packets sent: ");
  Serial.println(packetCount);
  
}

// put function definitions here:

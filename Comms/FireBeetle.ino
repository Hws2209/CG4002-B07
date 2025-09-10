#include <Arduino.h>
#include <WiFi.h>

//WiFi setup
const char* ssid = "Wenwuuu";
const char* password = "11223344";
const char* host = "10.104.169.64";
const int port = 2105;
int count = 0;

WiFiClient client;

int16_t ax, ay, az, gx, gy, gz, mx, my, mz;

struct SensorPacket {
  uint16_t header;
  uint16_t device_id; //denote left/right arm/leg, use 2 bytes for alignment
  uint16_t ax, ay, az;
  uint16_t gx, gy, gz; 
  uint16_t mx, my, mz;
};

SensorPacket packet;

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
  uint8_t mac[6];
  WiFi.macAddress(mac);
  packet.device_id = (mac[5] % 4)+1;
  Serial.print("Device ID: "); Serial.println(packet.device_id);

  //Connect to laptop
  if (!client.connect(host,port)){
    Serial.println("Connection Failed");
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
  packet.ax = 0xAB01;
  packet.ay = 0xCD10;
  packet.az = 0xEF11;
  packet.gx = 0xAB02;
  packet.gy = 0xCD20;
  packet.gz = 0xEF22;
  packet.mx = 0xAB03;
  packet.my = 0xCD30;
  packet.mz = 0xEF33;
  
}

void loop() {
  while(client.available()==0){
    delay(10); //wait for server to reply
  }

  String reply = client.readStringUntil('\n');
  Serial.println("Reply from server: " + reply);
  //client.write(mx); //sends only 1 byte. supposedly faster but not sure by how much
  //client.print(mx);
  client.write((uint8_t*)&packet, sizeof(packet)); // 22 bytes
  //count += 1;
}

// put function definitions here:
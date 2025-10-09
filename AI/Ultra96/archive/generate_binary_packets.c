#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#pragma pack(push, 1)  // ensure no padding in struct
struct SensorPacket {
    uint16_t header;
    uint16_t device_id;
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
};
#pragma pack(pop)

int main(void) {
    FILE *fin = fopen("data.txt", "r");
    FILE *fbin = fopen("packets.bin", "wb"); // binary write
    if (!fin || !fbin) {
        perror("File open error");
        return 1;
    }

    struct SensorPacket packet;
    packet.header = 0xFF;
    int ax, ay, az, gx, gy, gz;
    int device_id = 1;

    char line[256];
    while (fgets(line, sizeof(line), fin)) {
        // skip empty lines
        if (strlen(line) <= 1) continue;

        // read 6 integers from line
        if (sscanf(line, "%d %d %d %d %d %d", &ax, &ay, &az, &gx, &gy, &gz) == 6) {
            packet.device_id = device_id;
            packet.ax = (int16_t)ax;
            packet.ay = (int16_t)ay;
            packet.az = (int16_t)az;
            packet.gx = (int16_t)gx;
            packet.gy = (int16_t)gy;
            packet.gz = (int16_t)gz;

            // write to binary file
            fwrite(&packet, sizeof(packet), 1, fbin);

            // loop device_id 1 → 4
            device_id++;
            if (device_id > 4) device_id = 1;
        }
    }

    fclose(fin);
    fclose(fbin);
    return 0;
}

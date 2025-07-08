#include<stdio.h>
#include<linux/types.h>
#include<fcntl.h>
#include<unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <unistd.h>

#define I2C_DEV_NAME_X "/dev/i2c-4"
#define I2C_DEV_NAME_Y "/dev/i2c-7"
#define MCP4725_ADDR 0x60
#define SAMPLE_COUNT 256
#define DELAY_US 1953 

unsigned int fd_x;
unsigned int fd_y;

int square_x[SAMPLE_COUNT] = {
    0,   65,  130, 195, 260, 325, 390, 455,  // 顶边: X从0→4095
    520, 585, 650, 715, 780, 845, 910, 975,
    1040,1105,1170,1235,1300,1365,1430,1495,
    1560,1625,1690,1755,1820,1885,1950,2015,
    2080,2145,2210,2275,2340,2405,2470,2535,
    2600,2665,2730,2795,2860,2925,2990,3055,
    3120,3185,3250,3315,3380,3445,3510,3575,
    3640,3705,3770,3835,3900,3965,4030,4095,

    4095,4095,4095,4095,4095,4095,4095,4095,  // 右边: X固定4095
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,

    4095,4030,3965,3900,3835,3770,3705,3640,  // 底边: X从4095→0
    3575,3510,3445,3380,3315,3250,3185,3120,
    3055,2990,2925,2860,2795,2730,2665,2600,
    2535,2470,2405,2340,2275,2210,2145,2080,
    2015,1950,1885,1820,1755,1690,1625,1560,
    1495,1430,1365,1300,1235,1170,1105,1040,
    975, 910, 845, 780, 715, 650, 585, 520,
    455, 390, 325, 260, 195, 130, 65,  0,

    0,   0,   0,   0,   0,   0,   0,   0,    // 左边: X固定0
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0
};

int square_y[SAMPLE_COUNT] = {
    4095,4095,4095,4095,4095,4095,4095,4095,  // 顶边: Y固定4095
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,
    4095,4095,4095,4095,4095,4095,4095,4095,

    4095,4030,3965,3900,3835,3770,3705,3640,  // 右边: Y从4095→0
    3575,3510,3445,3380,3315,3250,3185,3120,
    3055,2990,2925,2860,2795,2730,2665,2600,
    2535,2470,2405,2340,2275,2210,2145,2080,
    2015,1950,1885,1820,1755,1690,1625,1560,
    1495,1430,1365,1300,1235,1170,1105,1040,
    975, 910, 845, 780, 715, 650, 585, 520,
    455, 390, 325, 260, 195, 130, 65,  0,

    0,   0,   0,   0,   0,   0,   0,   0,    // 底边: Y固定0
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,

    0,   65,  130, 195, 260, 325, 390, 455,  // 左边: Y从0→4095
    520, 585, 650, 715, 780, 845, 910, 975,
    1040,1105,1170,1235,1300,1365,1430,1495,
    1560,1625,1690,1755,1820,1885,1950,2015,
    2080,2145,2210,2275,2340,2405,2470,2535,
    2600,2665,2730,2795,2860,2925,2990,3055,
    3120,3185,3250,3315,3380,3445,3510,3575,
    3640,3705,3770,3835,3900,3965,4030,4095
};

int mcp4725_write_dac(int fd, unsigned char devaddr, unsigned short dac_value, int save_to_eeprom)
{
    struct i2c_rdwr_ioctl_data work_queue;
    unsigned char buffer[3]; // 控制字节 + 高字节 + 低字节
    int ret;
    
    buffer[0] = 0x40; // 010 00 000: 写入DAC寄存器，不保存到EEPROM，PD=00
    if (save_to_eeprom) {
        buffer[0] = 0x60; // 011 00 000: 写入DAC寄存器并保存到EEPROM
    }

    buffer[1] = (dac_value >> 4) & 0xFF; // 高8位（D11-D4）
    buffer[2] = (dac_value & 0x0F) << 4; // 低4位左移4位（D3-D0 → 高4位）

    work_queue.nmsgs = 1;
    work_queue.msgs = (struct i2c_msg *)malloc(work_queue.nmsgs * sizeof(work_queue.msgs));
    if (!work_queue.msgs) {
        printf("memory alloc failed");
        ret =-1;
        return ret;
    }

    work_queue.nmsgs = 1;
    work_queue.msgs[0].addr = devaddr;     // 设备地址
    work_queue.msgs[0].flags = 0;          // 写标志
    work_queue.msgs[0].len = sizeof(buffer);
    work_queue.msgs[0].buf = (unsigned char*)malloc(sizeof(buffer));
    if(!work_queue.msgs[0].buf) {
        perror("Failed to allocate buffer");
        free(work_queue.msgs);
        return -1;
    }
    memcpy(work_queue.msgs[0].buf, buffer, sizeof(buffer)); //你要写的数据

    ret = ioctl(fd, I2C_RDWR, (unsigned long)&work_queue);
    if (ret < 0)
        printf("error during I2C_RDWR ioctl with error code %d\n", ret);

    free((work_queue.msgs[0]).buf);
    free(work_queue.msgs);
    return ret;
}

int main()
{
    int ret;
    fd_x=open(I2C_DEV_NAME_X,O_RDWR);
    fd_y=open(I2C_DEV_NAME_Y,O_RDWR);
    if (fd_x < 0 || fd_y < 0) {
        perror("open i2c device failed");
        exit(EXIT_FAILURE);
    }


    while(1) {
        for(int i = 0; i < SAMPLE_COUNT; i++) {
            // 写入X轴
            ret = mcp4725_write_dac(fd_x, MCP4725_ADDR, square_x[i] & 0x0FFF, 0);
            // 写入Y轴
            ret |= mcp4725_write_dac(fd_y, MCP4725_ADDR, square_y[i] & 0x0FFF, 0);
            if(ret < 0) {
                fprintf(stderr, "Failed to write DAC at sample %d\n", i);
                close(fd_x);
                close(fd_y);
                exit(EXIT_FAILURE);
            }
            
            // 控制输出频率
            usleep(DELAY_US);
        }
    }
    close(fd_x);
    close(fd_y);
    return 0;
}
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
#include <math.h>
#include <sys/stat.h>
#include <time.h> // 添加时间处理头文件

#define I2C_DEV_NAME_X "/dev/i2c-4"
#define I2C_DEV_NAME_Y "/dev/i2c-7"
#define MCP4725_ADDR 0x60
#define DELAY_US 1953 
#define X_FILE_PATH "/home/elf/myyolo/logs/x_origin.log"
#define Y_FILE_PATH "/home/elf/myyolo/logs/y_origin.log"
#define SCRIPT_PATH_ON "./light_on.sh"
#define SCRIPT_PATH_OFF "./light_down.sh"
#define INACTIVITY_TIMEOUT 5 // 5秒超时

/* 执行脚本的函数 */
void execute_script(const char *script_path) {
    pid_t pid = fork();
    if (pid == 0) {
        execl(script_path, script_path, (char *)NULL);
        perror("execl failed");
        exit(EXIT_FAILURE);
    } else if (pid < 0) {
        perror("fork failed");
    }
}

unsigned int fd_x;
unsigned int fd_y;

int read_single_value(const char *filename, int *value) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open file");
        return -1;
    }

    if (fscanf(fp, "%d", value) != 1) {
        fprintf(stderr, "Invalid data in %s\n", filename);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

int mcp4725_write_dac(int fd, unsigned char devaddr, unsigned short dac_value, int save_to_eeprom) {
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

int main() {
    int ret;
    int x_value = 0, y_value = 0;
    struct stat last_x_stat, last_y_stat;
    time_t last_update_time; // 记录最后更新时间
    time_t current_time;     // 当前时间
    int inactivity_detected = 0; // 超时检测标志

    // 获取当前时间作为初始时间
    time(&last_update_time);

    fd_x = open(I2C_DEV_NAME_X, O_RDWR);
    fd_y = open(I2C_DEV_NAME_Y, O_RDWR);
    if (fd_x < 0 || fd_y < 0) {
        perror("open i2c device failed");
        exit(EXIT_FAILURE);
    }

    if (read_single_value(X_FILE_PATH, &x_value) != 0 ||
        read_single_value(Y_FILE_PATH, &y_value) != 0) {
        close(fd_x);
        close(fd_y);
        exit(EXIT_FAILURE);
    }
    // Convert initial values
    //最后面的x和y
    // x_value = round(23.67 * x_value - 6391.04);
    // x_value = (x_value > 4095) ? 4095 : x_value;
    // y_value = round(22.38 * y_value - 6645.98);
    // y_value = (y_value > 4095) ? 4095 : y_value;
    x_value = round(21.90 * x_value - 5605.90);
    x_value = (x_value > 4095) ? 4095 : x_value;
    y_value = round(23.41 * y_value - 8239);
    y_value = (y_value > 2996) ? 2996 : y_value;
    stat(X_FILE_PATH, &last_x_stat);
    stat(Y_FILE_PATH, &last_y_stat);

    while(1) {
        int data_updated = 0;  // 数据更新标志

        // 检查文件是否更新
        struct stat current_x_stat, current_y_stat;
        if (stat(X_FILE_PATH, &current_x_stat) == 0 && 
            current_x_stat.st_mtime != last_x_stat.st_mtime) {
            if (read_single_value(X_FILE_PATH, &x_value) == 0) {
                int real_x_value = round(21.90 * x_value - 5605.90);
                x_value = (real_x_value > 4095) ? 4095 : real_x_value;
                last_x_stat = current_x_stat;
                printf("Updated X value: %d\n", x_value);
                data_updated = 1;  // 标记数据已更新
            }
        }
        if (stat(Y_FILE_PATH, &current_y_stat) == 0 && 
            current_y_stat.st_mtime != last_y_stat.st_mtime) {
            if (read_single_value(Y_FILE_PATH, &y_value) == 0) {
                int real_y_value = round(23.41 * y_value - 8239);
                y_value = (real_y_value > 2996) ? 2996 : real_y_value;
                last_y_stat = current_y_stat;
                printf("Updated Y value: %d\n", y_value);
                data_updated = 1;  // 标记数据已更新
            }
        }

        // 如果数据变化，更新最后活动时间并执行开灯脚本
        if (data_updated) {
            time(&last_update_time); // 更新最后活动时间
            printf("Data updated, executing on script...\n");
            execute_script(SCRIPT_PATH_ON);
            inactivity_detected = 0; // 重置超时标志
        }
        
        // 检查是否超时（10秒无更新）
        time(&current_time);
        double seconds_since_update = difftime(current_time, last_update_time);
        
        if (seconds_since_update >= INACTIVITY_TIMEOUT) {
            if (!inactivity_detected) {
                printf("No update for %.2f seconds, executing off script...\n", seconds_since_update);
                execute_script(SCRIPT_PATH_OFF);
                inactivity_detected = 1; // 设置标志避免重复执行
            }
        } else {
            inactivity_detected = 0; // 重置标志
        }

        // 写入X轴
        ret = mcp4725_write_dac(fd_x, MCP4725_ADDR, x_value & 0x0FFF, 0);
        // 写入Y轴
        ret |= mcp4725_write_dac(fd_y, MCP4725_ADDR, y_value & 0x0FFF, 0);
        if(ret < 0) {
            fprintf(stderr, "DAC write error\n");
            close(fd_x);
            close(fd_y);
            exit(EXIT_FAILURE);
        }
            
        // 控制输出频率
        usleep(DELAY_US);        
    }
    close(fd_x);
    close(fd_y);
    return 0;
}
#!/bin/bash

# PWM 配置脚本（简易版）
PWM_PATH="/sys/class/pwm/pwmchip0/pwm0"

if [ ! -d "$PWM_PATH" ]; then
    echo "导出PWM通道0..."
    echo 0 > /sys/class/pwm/pwmchip0/export
fi
  
echo "设置PWM周期..."
echo 1000000 > $PWM_PATH/period

echo "设置PWM占空比..."
echo 650000 > $PWM_PATH/duty_cycle

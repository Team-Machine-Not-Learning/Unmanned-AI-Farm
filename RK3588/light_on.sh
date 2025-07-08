#!/bin/bash

#关灯
LIGHT_PATH="/sys/class/gpio/gpio107"

if [ ! -d "$LIGHT_PATH" ]; then
    echo 107 > /sys/class/gpio/export
fi

echo out > $LIGHT_PATH/direction
echo 1 > $LIGHT_PATH/value

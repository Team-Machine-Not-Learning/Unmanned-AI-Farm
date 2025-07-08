#!/bin/bash

#继电器开
RELAY_PATH="/sys/class/gpio/gpio109"

if [ ! -d "$RELAY_PATH" ]; then
    echo 109 > /sys/class/gpio/export
fi

echo out > $RELAY_PATH/direction
echo 1 > $RELAY_PATH/value

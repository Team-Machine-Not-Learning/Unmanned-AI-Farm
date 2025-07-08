#!/bin/bash

#水阀开
FEED_VALUE_PATH="/sys/class/gpio/gpio108"

if [ ! -d "$FEED_VALUE_PATH" ]; then
    echo 108 > /sys/class/gpio/export
fi

echo out > $FEED_VALUE_PATH/direction
echo 1 > $FEED_VALUE_PATH/value

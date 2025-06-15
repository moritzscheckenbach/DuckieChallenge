#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch image capture node in the background
rosrun image_capture image_capture.py &
IMAGE_CAPTURE_PID=$!

echo "Bild-Aufnahme-Node wurde gestartet. Warte auf Nutzer-Eingaben..."
echo "Drücke [p], um ein Bild aufzunehmen, oder [q] zum Beenden."

# Launch trigger node with interactive terminal
dt-exec rosrun image_capture image_trigger.py

# Kill the background process when the foreground ends
kill $IMAGE_CAPTURE_PID

# wait for app to end
dt-launchfile-join
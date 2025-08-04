#!/bin/bash
set -e

echo " Checking and freeing required ports..."

# Ports in use 
JAVA_PORT=8080
FASTAPI_PORT=8000
REACT_PORT=3000

#  Function to kill process using a port
kill_port() {
  local PORT=$1
  PID=$(lsof -t -i:$PORT || true)
  if [ -n "$PID" ]; then
    echo " Port $PORT is in use by PID $PID, killing..."
    kill -9 $PID
  else
    echo "Port $PORT is free"
  fi
}

# STEP 1: Load DB credentials
echo "Loading DB credentials..."
chmod +x ./data_base_cred.sh
./data_base_cred.sh

# STEP 2: Free ports
kill_port $JAVA_PORT
kill_port $FASTAPI_PORT
kill_port $REACT_PORT

echo "Ports are free. Starting compilation..."

# STEP 3: Compile Java Backend 
echo " Compiling Java backend files..."
javac -cp ".:Data_Base:Data_Base/postgresql-42.7.3.jar:Data_Base/json.jar" \
  Backend/SpamCollectorServer.java Data_Base/SpamDatabase.java

if [ $? -ne 0 ]; then
  echo " Compilation failed! Check for errors."
  exit 1
else
  echo " Compilation successful!"
fi

echo " Starting all services..."

# === STEP 4: Start Java Backend
java -cp ".:Data_Base:Data_Base/postgresql-42.7.3.jar:Data_Base/json.jar" \
  Backend.SpamCollectorServer &
echo " Java Backend started on port $JAVA_PORT"

#  STEP 5: Start FastAPI Backend 
uvicorn Backend.project_FastApi:app --host 0.0.0.0 --port $FASTAPI_PORT &
echo " FastAPI Server started on port $FASTAPI_PORT"

#  STEP 6: Start React Frontend
cd Frontend/spam-detector-ui
npm start &
echo "React Frontend started on port $REACT_PORT"

cd /app
echo "All services are running!"

# STEP 7: Keep container alive
wait

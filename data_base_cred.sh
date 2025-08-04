#!/bin/bash
set -e

CONFIG_FILE="data_base_config.txt"

#  Load DB config from data_base_config.txt
echo " Loading DB configuration..."
while IFS='=' read -r key value; do
  # Skip empty lines or comments
  if [[ -n "$key" && ! "$key" =~ ^# ]]; then
    export "$key"="$value"
    echo "  Loaded $key=$value"
  fi
done < "$CONFIG_FILE"

# Combine DB_HOST_URL + DB_NAME → DB_URL
export DB_URL="${DB_HOST_URL}${DB_NAME}"
echo "Final DB_URL=$DB_URL"

# Compile SpamDatabase.java
echo "🛠 Compiling SpamDatabase.java..."
javac -cp ".:postgresql-42.7.3.jar:json.jar" SpamDatabase.java

# TEST DB connection interactively (optional)
echo "Testing DB connection..."
java -cp ".:postgresql-42.7.3.jar:json.jar" \
    -DDB_URL="$DB_URL" \
    -DDB_USER="$DB_USER" \
    SpamDatabase

# Start Java Backend
echo "Starting Java Backend..."
javac -cp .:../Data_Base:../Data_Base/postgresql-42.7.3.jar:../Data_Base/json.jar ../Backend/SpamCollectorServer.java
java -cp .:../Data_Base:../Data_Base/postgresql-42.7.3.jar:../Data_Base/json.jar Backend.SpamCollectorServer &
echo "Java Backend started..."

# Keep running
wait

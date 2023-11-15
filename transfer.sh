#!/bin/bash

# Function to display usage information
function show_usage {
  echo "Usage: $0 [-g|--get|-p|--put] <source_directory>"
  exit 1
}

# Function to prompt for a password
function prompt_for_password {
  read -s -p "Enter password for Orion: " ORION_PASSWORD
  echo
}

# Check if an option and source directory are provided
if [ "$#" -lt 2 ]; then
  show_usage
fi

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -g|--get)
      OPERATION="get"
      ;;
    -p|--put)
      OPERATION="put"
      ;;
    *)
      SOURCE_DIR="$1"
      ;;
  esac
  shift
done

# Check if a valid operation is specified
if [ -z "$OPERATION" ]; then
  show_usage
fi

# Define variables
USER="parlab19"
SOURCE_HOST="scirouter"
DEST_HOST="orion"
DEST_DIR="~/"

# Prompt for Orion password
prompt_for_password

function pass {
  sshpass -p "$ORION_PASSWORD"
}

# Perform the specified operation
case $OPERATION in
  "get")
    echo "Fetching $SOURCE_DIR from scirouter to orion"

    # pass from source host to destination host
    sshpass -p "$ORION_PASSWORD" scp -r -o ProxyJump=$USER@$DEST_HOST $USER@$SOURCE_HOST:~/pps/$SOURCE_DIR $USER@$DEST_HOST:$DEST_DIR

    echo "Fetching $SOURCE_DIR from orion"

    # pass from destination host to local host
    sshpass -p "$ORION_PASSWORD" scp -r $USER@$DEST_HOST:$DEST_DIR$SOURCE_DIR .

    echo done

    ;;
  "put")
    # pass from local host to destination host
    sshpass -p "$ORION_PASSWORD" scp -r $SOURCE_DIR $USER@$DEST_HOST:$DEST_DIR

    # pass from destination host to source host
    sshpass -p "$ORION_PASSWORD" scp -r -o ProxyJump=$USER@$DEST_HOST $USER@$SOURCE_HOST:$DEST_DIR $USER@$SOURCE_HOST:~/pps/
    ;;
  *)
    show_usage
    ;;
esac

echo "Cleaning up orion"

# Remove files from destination host after pass
sshpass -p "$ORION_PASSWORD" ssh $USER@$DEST_HOST "rm -rf $DEST_DIR/$SOURCE_DIR"

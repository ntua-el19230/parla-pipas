#!/bin/bash

# Function to display usage information
function show_usage {
  echo "Usage: $0 [-g|--get|-p|--put] <source_directory>"
  exit 1
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

# Perform the specified operation
case $OPERATION in
  "get")
    # Transfer from source host to destination host
    scp -r -o ProxyJump=$USER@$DEST_HOST $USER@$SOURCE_HOST:~/pps/$SOURCE_DIR $USER@$DEST_HOST:$DEST_DIR

    # Transfer from destination host to local host
    scp -r $USER@$DEST_HOST:$DEST_DIR/$SOURCE_DIR .
    ;;
  "put")
    # Transfer from local host to destination host
    scp -r $SOURCE_DIR $USER@$DEST_HOST:$DEST_DIR

    # Transfer from destination host to source host
    scp -r -o ProxyJump=$USER@$DEST_HOST $USER@$SOURCE_HOST:$DEST_DIR $USER@$SOURCE_HOST:~/pps/
    ;;
  *)
    show_usage
    ;;
esac

# Remove files from destination host after transfer
ssh $USER@$DEST_HOST "rm -r $DEST_DIR"

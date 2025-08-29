#!/bin/bash
set -e

CACHE_DIR=/tmp/vectorstore

# Create cache directory
mkdir -p $CACHE_DIR

# Copy cached vectorstore if exists, else pull from Git LFS
if [ -d $CACHE_DIR ] && [ "$(ls -A $CACHE_DIR)" ]; then
    echo "Using cached vectorstore"
    cp -r $CACHE_DIR/* app/vectorstore/
else
    echo "No cache found, pulling LFS data"
    git lfs pull
    cp -r app/vectorstore/* $CACHE_DIR/
fi

# Install Python dependencies
python -m pip install -r requirements.txt

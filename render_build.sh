#!/bin/bash
set -e

# Create cache directory
mkdir -p /var/cache/vectorstore

# Copy cached vectorstore if exists, else pull from Git LFS
if [ -d /var/cache/vectorstore ] && [ "$(ls -A /var/cache/vectorstore)" ]; then
    echo "Using cached vectorstore"
    cp -r /var/cache/vectorstore/* app/vectorstore/
else
    echo "No cache found, pulling LFS data"
    git lfs pull
    cp -r app/vectorstore/* /var/cache/vectorstore/
fi

# Install Python dependencies
python -m pip install -r requirements.txt

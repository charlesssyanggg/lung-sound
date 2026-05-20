#!/bin/bash
# Download ICBHI 2017 Dataset from Kaggle
# Requires: kaggle CLI configured with API key
# See: https://github.com/Kaggle/kaggle-api

echo "Downloading ICBHI 2017 Respiratory Sound Database..."
mkdir -p data
kaggle datasets download -d vbookshelf/respiratory-sound-database -p data/
unzip data/respiratory-sound-database.zip -d data/ICBHI_2017
echo "Done. Data saved to: data/ICBHI_2017/"

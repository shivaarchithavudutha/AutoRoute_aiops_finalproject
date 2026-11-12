#!/bin/sh
# Downloads the CFPB narrative exports used in the project (version-controlled data collection script).
# Usage: sh scripts/download_data.sh            (training files)
#        sh scripts/download_data.sh production  (newer files for the drift demo)
# If a link fails, download manually from:
# https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
set -e
BASE=https://files.consumerfinance.gov/f/documents
if [ "$1" = "production" ]; then
  DIR=data/raw/production
  FILES="CCDB_Export_9_January_2025_through_February_2025.zip"
else
  DIR=data/raw/train
  FILES="CCDB_Export_2_May_2018_through_April_2021.zip CCDB_Export_3_May_2021_through_October_2022.zip"
fi
mkdir -p "$DIR"
for f in $FILES; do
  echo "Downloading $f"
  wget -c -P "$DIR" "$BASE/$f"
  unzip -o -d "$DIR" "$DIR/$f"
  rm "$DIR/$f"
done
echo "Done. Files in $DIR:"
ls -lh "$DIR"

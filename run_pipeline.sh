#!/bin/bash
set -e

echo "=== SurgeryAudit Data Pipeline ==="
echo "Started: $(date)"

DATA_DIR="/app/data"
DATASET_DIR="/app/dataset"
LIMIT=${SCRAPE_LIMIT:-300}

echo ""
echo "=== Step 1: Scrape Reddit (real surgery photos) ==="
python scrape_reddit.py --real-only --limit $LIMIT --output-dir $DATA_DIR --sort top --time-filter all

echo ""
echo "=== Step 2: Scrape Reddit (manipulated/filtered photos) ==="
python scrape_reddit.py --manipulated-only --limit $LIMIT --output-dir $DATA_DIR --sort top --time-filter all

echo ""
echo "=== Step 3: Generate synthetic manipulations ==="
python generate_synthetic.py --input-dir $DATA_DIR/real --output-dir $DATA_DIR/synthetic --method all

echo ""
echo "=== Step 4: Build balanced dataset ==="
python build_dataset.py --data-dir $DATA_DIR --output-dir $DATASET_DIR --target-size 512

echo ""
echo "=== Step 5: Upload results to GCS ==="
if [ -n "$GCS_BUCKET" ]; then
    apt-get update -qq && apt-get install -y -qq google-cloud-cli > /dev/null 2>&1 || true
    gsutil -m cp -r $DATASET_DIR/* gs://$GCS_BUCKET/dataset/
    gsutil -m cp -r $DATA_DIR/metadata/* gs://$GCS_BUCKET/metadata/
    echo "Uploaded to gs://$GCS_BUCKET/"
else
    echo "No GCS_BUCKET set, skipping upload. Data at $DATASET_DIR"
fi

echo ""
echo "=== Pipeline complete ==="
echo "Finished: $(date)"
ls -lh $DATASET_DIR/
echo ""
echo "Dataset stats:"
cat $DATASET_DIR/stats.json 2>/dev/null || echo "No stats file"

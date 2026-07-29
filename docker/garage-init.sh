#!/bin/sh
# Initialize Garage S3 bucket and credentials for Langfuse v3
# Run once on startup to set up the bucket and generate keys

set -e

GARAGE_ADMIN_URL="http://garage:3903"
GARAGE_S3_ENDPOINT="http://garage:3900"
BUCKET_NAME="langfuse-media"
KEY_NAME="langfuse-key"
KEYS_VOLUME="/garage-keys"

echo "[Garage Init] Waiting for Garage admin API..."
MAX_ATTEMPTS=30
ATTEMPT=0
while ! nc -z garage 3903 2>/dev/null; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
        echo "[Garage Init] ERROR: Garage admin API did not become ready after ${MAX_ATTEMPTS} attempts"
        exit 1
    fi
    echo "[Garage Init] Attempt $ATTEMPT/$MAX_ATTEMPTS..."
    sleep 1
done

echo "[Garage Init] Garage admin API is ready"

# Get the node ID
echo "[Garage Init] Getting node ID..."
NODE_ID=$(curl -s "$GARAGE_ADMIN_URL/status" | grep -o '"id":"[^"]*' | cut -d'"' -f4 || echo "")
if [ -z "$NODE_ID" ]; then
    echo "[Garage Init] ERROR: Could not determine node ID"
    exit 1
fi
echo "[Garage Init] Node ID: $NODE_ID"

# Assign layout
echo "[Garage Init] Assigning layout..."
curl -s -X POST "$GARAGE_ADMIN_URL/layout/assign" \
    -H "Content-Type: application/json" \
    -d "{\"id\":\"$NODE_ID\",\"zone\":\"dc1\",\"capacity_gb\":10}" || true

# Apply layout
echo "[Garage Init] Applying layout..."
curl -s -X POST "$GARAGE_ADMIN_URL/layout/apply" \
    -H "Content-Type: application/json" \
    -d '{"version":1}' || true

sleep 2

# Create bucket
echo "[Garage Init] Creating bucket '$BUCKET_NAME'..."
BUCKET_RESPONSE=$(curl -s -X POST "$GARAGE_ADMIN_URL/buckets" \
    -H "Content-Type: application/json" \
    -d "{\"globalAlias\":\"$BUCKET_NAME\"}")
echo "[Garage Init] Bucket response: $BUCKET_RESPONSE"

# Create key
echo "[Garage Init] Creating key '$KEY_NAME'..."
KEY_RESPONSE=$(curl -s -X POST "$GARAGE_ADMIN_URL/keys" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$KEY_NAME\"}")
echo "[Garage Init] Key response: $KEY_RESPONSE"

# Extract key ID and secret from response
KEY_ID=$(echo "$KEY_RESPONSE" | grep -o '"id":"[^"]*' | head -1 | cut -d'"' -f4 || echo "")
KEY_SECRET=$(echo "$KEY_RESPONSE" | grep -o '"secret_key":"[^"]*' | cut -d'"' -f4 || echo "")

if [ -z "$KEY_ID" ] || [ -z "$KEY_SECRET" ]; then
    echo "[Garage Init] ERROR: Could not extract key ID or secret"
    echo "[Garage Init] Response was: $KEY_RESPONSE"
    exit 1
fi

echo "[Garage Init] Key ID: $KEY_ID"
echo "[Garage Init] Key Secret: (redacted)"

# Grant read/write access to bucket
echo "[Garage Init] Granting access to bucket..."
curl -s -X POST "$GARAGE_ADMIN_URL/keys/$KEY_ID/allow" \
    -H "Content-Type: application/json" \
    -d "{\"bucketName\":\"$BUCKET_NAME\",\"permissions\":{\"read\":true,\"write\":true}}" || true

# Write credentials to shared volume for langfuse services
mkdir -p "$KEYS_VOLUME"
cat > "$KEYS_VOLUME/s3-credentials" << EOF
LANGFUSE_S3_ACCESS_KEY_ID=$KEY_ID
LANGFUSE_S3_SECRET_ACCESS_KEY=$KEY_SECRET
EOF

chmod 600 "$KEYS_VOLUME/s3-credentials"
echo "[Garage Init] Wrote credentials to $KEYS_VOLUME/s3-credentials"

echo "[Garage Init] ✓ Garage initialization complete"

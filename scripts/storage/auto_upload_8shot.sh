#!/bin/bash
# Auto-upload 8-shot trajectories to B2 when collection completes

cd ~/p1_geom_characterization

# Load B2 credentials
source <(grep -E "^B2_" configs/b2-configs.txt | sed 's/ //g')

UPLOAD_PREFIX="trajectories_8shot_logiqa_$(date +%Y%m%d)"
LOCAL_DIR="data/trajectories_8shot"

echo "=============================================="
echo "Auto-upload to B2: $UPLOAD_PREFIX"
echo "=============================================="

# Function to upload a file
upload_file() {
    local filepath=$1
    local remote_name=$(basename $filepath)
    local model_dir=$(basename $(dirname $filepath))

    echo "Uploading: $model_dir/$remote_name"
    python3 -c "
from b2sdk.v2 import InMemoryAccountInfo, B2Api
import os

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account(\"production\", os.environ[\"B2_KEY_ID\"], os.environ[\"B2_APP_KEY\"])
bucket = b2_api.get_bucket_by_name(os.environ[\"B2_BUCKET_NAME\"])

filepath = \"$filepath\"
remote_path = \"$UPLOAD_PREFIX/$model_dir/$remote_name\"
bucket.upload_local_file(filepath, remote_path)
print(f\"  Uploaded to: {remote_path}\")
"
}

export -f upload_file
export B2_KEY_ID B2_APP_KEY B2_BUCKET_NAME UPLOAD_PREFIX

# Find and upload all completed .h5 files
for model_dir in olmo3_base olmo3_sft olmo3_rl_zero olmo3_think; do
    h5_file="$LOCAL_DIR/$model_dir/logiqa_trajectories_8shot.h5"
    if [ -f "$h5_file" ]; then
        size=$(du -h "$h5_file" | cut -f1)
        echo "Found: $h5_file ($size)"
        upload_file "$h5_file"
    fi
done

echo ""
echo "Upload complete!"
echo "Files available at: b2://$B2_BUCKET_NAME/$UPLOAD_PREFIX/"

#!/bin/bash
# Script to copy fed_lfc_cbm repo to Nautilus pod
# Usage: ./copy_to_nautilus.sh <pod-name>
# Example: ./copy_to_nautilus.sh medcbm

POD_NAME=${1:-medcbm}
NAMESPACE="wenglab-interpretable-ai"
REPO_DIR=$(dirname "$(readlink -f "$0")")
TAR_FILE="/tmp/fed_lfc_cbm_$(date +%s).tar.gz"

echo "Creating archive of fed_lfc_cbm..."
cd "$REPO_DIR"

tar --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='saved_models' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='.DS_Store' \
    --exclude='venv' \
    --exclude='env' \
    -czf "$TAR_FILE" .

echo "Copying to pod $POD_NAME..."
kubectl cp "$TAR_FILE" "$NAMESPACE/$POD_NAME:/sc-cbint-vol/fed_lfc_cbm.tar.gz"

echo "Extracting in pod..."
kubectl exec -it "$POD_NAME" -n "$NAMESPACE" -- bash -c "cd /sc-cbint-vol && rm -rf fed_lfc_cbm && mkdir -p fed_lfc_cbm && tar xzf fed_lfc_cbm.tar.gz -C fed_lfc_cbm && rm fed_lfc_cbm.tar.gz && echo 'Extraction complete!'"

echo "Cleaning up local tar..."
rm "$TAR_FILE"

echo "Done! Repository copied to /sc-cbint-vol/fed_lfc_cbm in pod"
echo "Next steps:"
echo "  kubectl exec -it $POD_NAME -n $NAMESPACE -- bash"
echo "  cd /sc-cbint-vol/fed_lfc_cbm"
echo "  bash setup_nautilus.sh"

#!/bin/bash
set -e

# ============================================================
# ReasonForge - Wallet Setup Script
# Creates owner, miner, and validator wallets using btcli
# ============================================================

WALLET_NAME="${WALLET_NAME:-owner}"

echo "============================================"
echo "  ReasonForge Wallet Setup"
echo "============================================"
echo ""

# Check if btcli is installed
if ! command -v btcli &> /dev/null; then
    echo "ERROR: btcli is not installed. Install it with:"
    echo "  pip install bittensor"
    exit 1
fi

# Create owner coldkey
echo "[1/3] Creating owner coldkey: ${WALLET_NAME}"
btcli wallet new_coldkey --wallet.name "${WALLET_NAME}" --no_prompt 2>/dev/null || \
    echo "  -> Coldkey '${WALLET_NAME}' already exists, skipping."

# Create miner hotkey
echo "[2/3] Creating miner hotkey: miner"
btcli wallet new_hotkey --wallet.name "${WALLET_NAME}" --wallet.hotkey miner --no_prompt 2>/dev/null || \
    echo "  -> Hotkey 'miner' already exists, skipping."

# Create validator hotkey
echo "[3/3] Creating validator hotkey: validator"
btcli wallet new_hotkey --wallet.name "${WALLET_NAME}" --wallet.hotkey validator --no_prompt 2>/dev/null || \
    echo "  -> Hotkey 'validator' already exists, skipping."

echo ""
echo "============================================"
echo "  Wallet setup complete!"
echo "============================================"
echo ""
echo "Wallets created:"
btcli wallet list --wallet.name "${WALLET_NAME}" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Fund the coldkey with TAO"
echo "  2. Register on a subnet: ./scripts/register_neurons.sh"

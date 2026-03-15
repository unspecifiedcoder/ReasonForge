#!/bin/bash
set -e

# ============================================================
# ReasonForge - Register Neurons Script
# Registers miner and validator UIDs on a subnet
# ============================================================

WALLET_NAME="${WALLET_NAME:-owner}"
NETUID="${NETUID:-1}"
SUBTENSOR_NETWORK="${SUBTENSOR_NETWORK:-test}"
SUBTENSOR_CHAIN_ENDPOINT="${SUBTENSOR_CHAIN_ENDPOINT:-}"

echo "============================================"
echo "  ReasonForge Neuron Registration"
echo "============================================"
echo ""

# Check if btcli is installed
if ! command -v btcli &> /dev/null; then
    echo "ERROR: btcli is not installed. Install it with:"
    echo "  pip install bittensor"
    exit 1
fi

# Build chain endpoint args
CHAIN_ARGS=""
if [ -n "${SUBTENSOR_CHAIN_ENDPOINT}" ]; then
    CHAIN_ARGS="--subtensor.chain_endpoint ${SUBTENSOR_CHAIN_ENDPOINT}"
fi

echo "Network:    ${SUBTENSOR_NETWORK}"
echo "Wallet:     ${WALLET_NAME}"
echo "Subnet UID: ${NETUID}"
echo ""

# Register miner
echo "[1/2] Registering miner on subnet ${NETUID}..."
btcli subnet register \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey miner \
    --netuid "${NETUID}" \
    --subtensor.network "${SUBTENSOR_NETWORK}" \
    ${CHAIN_ARGS} \
    --no_prompt

echo ""

# Register validator
echo "[2/2] Registering validator on subnet ${NETUID}..."
btcli subnet register \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey validator \
    --netuid "${NETUID}" \
    --subtensor.network "${SUBTENSOR_NETWORK}" \
    ${CHAIN_ARGS} \
    --no_prompt

echo ""
echo "============================================"
echo "  Neuron registration complete!"
echo "============================================"
echo ""
echo "Verify registrations:"
echo "  btcli subnet metagraph --netuid ${NETUID} --subtensor.network ${SUBTENSOR_NETWORK} ${CHAIN_ARGS}"

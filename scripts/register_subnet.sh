#!/bin/bash
set -e

# ============================================================
# ReasonForge - Register Subnet Script
# Registers a new subnet on testnet
# ============================================================

WALLET_NAME="${WALLET_NAME:-owner}"
SUBTENSOR_NETWORK="${SUBTENSOR_NETWORK:-test}"
SUBTENSOR_CHAIN_ENDPOINT="${SUBTENSOR_CHAIN_ENDPOINT:-}"

echo "============================================"
echo "  ReasonForge Subnet Registration"
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
echo ""

# Register subnet
echo "[1/1] Registering new subnet on ${SUBTENSOR_NETWORK}..."
btcli subnet create \
    --wallet.name "${WALLET_NAME}" \
    --subtensor.network "${SUBTENSOR_NETWORK}" \
    ${CHAIN_ARGS} \
    --no_prompt

echo ""
echo "============================================"
echo "  Subnet registration complete!"
echo "============================================"
echo ""
echo "List subnets to find your NETUID:"
echo "  btcli subnet list --subtensor.network ${SUBTENSOR_NETWORK} ${CHAIN_ARGS}"
echo ""
echo "Next steps:"
echo "  1. Note your NETUID"
echo "  2. Register neurons: NETUID=<your_netuid> ./scripts/register_neurons.sh"

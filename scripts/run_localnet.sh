#!/bin/bash
set -e

# ============================================================
# ReasonForge - Local Network Setup Script
# Starts local subtensor, creates wallets, funds them,
# registers subnet, registers neurons, and stakes
# ============================================================

WALLET_NAME="${WALLET_NAME:-owner}"
STAKE_AMOUNT="${STAKE_AMOUNT:-1000}"
SUBTENSOR_CHAIN_ENDPOINT="${SUBTENSOR_CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"

echo "============================================"
echo "  ReasonForge Local Network Setup"
echo "============================================"
echo ""

# Check prerequisites
for cmd in btcli docker; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "ERROR: ${cmd} is not installed."
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# ---- Step 1: Start local subtensor ----
echo "[1/6] Starting local subtensor..."
docker run -d \
    --name subtensor-local \
    --rm \
    -p 9944:9944 \
    -p 9933:9933 \
    -p 30333:30333 \
    opentensor/subtensor:latest \
    --dev \
    --ws-external \
    --rpc-external \
    --rpc-cors all 2>/dev/null || echo "  -> subtensor-local already running."

echo "  Waiting for subtensor to be ready..."
sleep 10

CHAIN_ARGS="--subtensor.chain_endpoint ${SUBTENSOR_CHAIN_ENDPOINT}"

# ---- Step 2: Create wallets ----
echo ""
echo "[2/6] Creating wallets..."
btcli wallet new_coldkey --wallet.name "${WALLET_NAME}" --no_prompt 2>/dev/null || \
    echo "  -> Coldkey '${WALLET_NAME}' already exists."
btcli wallet new_hotkey --wallet.name "${WALLET_NAME}" --wallet.hotkey miner --no_prompt 2>/dev/null || \
    echo "  -> Hotkey 'miner' already exists."
btcli wallet new_hotkey --wallet.name "${WALLET_NAME}" --wallet.hotkey validator --no_prompt 2>/dev/null || \
    echo "  -> Hotkey 'validator' already exists."

# ---- Step 3: Fund wallets from faucet ----
echo ""
echo "[3/6] Funding wallets from faucet..."
btcli wallet faucet \
    --wallet.name "${WALLET_NAME}" \
    --subtensor.network local \
    ${CHAIN_ARGS} \
    --no_prompt 2>/dev/null || echo "  -> Faucet funding attempted (may require manual funding)."

# ---- Step 4: Register subnet ----
echo ""
echo "[4/6] Registering subnet..."
btcli subnet create \
    --wallet.name "${WALLET_NAME}" \
    --subtensor.network local \
    ${CHAIN_ARGS} \
    --no_prompt 2>/dev/null || echo "  -> Subnet may already exist."

NETUID=1
echo "  Using NETUID=${NETUID}"

# ---- Step 5: Register neurons ----
echo ""
echo "[5/6] Registering neurons on subnet ${NETUID}..."
btcli subnet register \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey miner \
    --netuid "${NETUID}" \
    --subtensor.network local \
    ${CHAIN_ARGS} \
    --no_prompt 2>/dev/null || echo "  -> Miner may already be registered."

btcli subnet register \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey validator \
    --netuid "${NETUID}" \
    --subtensor.network local \
    ${CHAIN_ARGS} \
    --no_prompt 2>/dev/null || echo "  -> Validator may already be registered."

# ---- Step 6: Stake TAO for validator ----
echo ""
echo "[6/6] Staking ${STAKE_AMOUNT} TAO for validator..."
btcli stake add \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey validator \
    --amount "${STAKE_AMOUNT}" \
    --subtensor.network local \
    ${CHAIN_ARGS} \
    --no_prompt 2>/dev/null || echo "  -> Staking attempted (may need more funds)."

echo ""
echo "============================================"
echo "  Local network setup complete!"
echo "============================================"
echo ""
echo "Subtensor:  ${SUBTENSOR_CHAIN_ENDPOINT}"
echo "Subnet:     ${NETUID}"
echo "Wallet:     ${WALLET_NAME}"
echo ""
echo "Start services:"
echo "  cd ${PROJECT_DIR}/docker"
echo "  docker compose -f docker-compose.localnet.yml up --build"
echo ""
echo "To stop local subtensor:"
echo "  docker stop subtensor-local"

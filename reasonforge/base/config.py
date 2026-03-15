"""
ReasonForge - Configuration Management

CLI argument parsing for miner and validator neurons.
Gracefully degrades when bittensor is not installed.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import bittensor as bt

    HAS_BITTENSOR = True
except ImportError:
    HAS_BITTENSOR = False


@dataclass
class MinerConfig:
    """Miner-specific configuration."""

    backend: str = "openai"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    max_concurrent: int = 4
    port: int = 8091
    domains: List[str] = field(
        default_factory=lambda: [
            "mathematics",
            "code",
            "scientific",
            "strategic",
            "causal",
            "ethical",
        ]
    )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Miner")
        group.add_argument(
            "--miner.backend",
            type=str,
            default="openai",
            choices=["openai", "anthropic", "local", "agent"],
            help="LLM backend to use",
        )
        group.add_argument("--miner.model", type=str, default="gpt-4o")
        group.add_argument("--miner.api_key_env", type=str, default="OPENAI_API_KEY")
        group.add_argument("--miner.max_concurrent", type=int, default=4)
        group.add_argument("--miner.port", type=int, default=8091)
        group.add_argument(
            "--miner.domains",
            type=str,
            nargs="+",
            default=["mathematics", "code", "scientific", "strategic", "causal", "ethical"],
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MinerConfig:
        return cls(
            backend=getattr(args, "miner.backend", "openai"),
            model=getattr(args, "miner.model", "gpt-4o"),
            api_key_env=getattr(args, "miner.api_key_env", "OPENAI_API_KEY"),
            max_concurrent=getattr(args, "miner.max_concurrent", 4),
            port=getattr(args, "miner.port", 8091),
            domains=getattr(args, "miner.domains", cls.domains),
        )

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


@dataclass
class ValidatorConfig:
    """Validator-specific configuration."""

    epoch_length: int = 360
    tasks_per_epoch: int = 12
    trap_rate: float = 0.15
    timeout: int = 300
    sample_size: int = 16
    port: int = 8092
    sandbox_enabled: bool = False
    lean4_enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Validator")
        group.add_argument(
            "--validator.epoch_length",
            type=int,
            default=360,
            help="Blocks per epoch (360 = ~72 min)",
        )
        group.add_argument("--validator.tasks_per_epoch", type=int, default=12)
        group.add_argument("--validator.trap_rate", type=float, default=0.15)
        group.add_argument("--validator.timeout", type=int, default=300)
        group.add_argument(
            "--validator.sample_size",
            type=int,
            default=16,
            help="Number of miners to query per task",
        )
        group.add_argument("--validator.port", type=int, default=8092)
        group.add_argument("--validator.sandbox_enabled", action="store_true")
        group.add_argument("--validator.lean4_enabled", action="store_true")
        group.add_argument(
            "--validator.embedding_model",
            type=str,
            default="all-MiniLM-L6-v2",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ValidatorConfig:
        return cls(
            epoch_length=getattr(args, "validator.epoch_length", 360),
            tasks_per_epoch=getattr(args, "validator.tasks_per_epoch", 12),
            trap_rate=getattr(args, "validator.trap_rate", 0.15),
            timeout=getattr(args, "validator.timeout", 300),
            sample_size=getattr(args, "validator.sample_size", 16),
            port=getattr(args, "validator.port", 8092),
            sandbox_enabled=getattr(args, "validator.sandbox_enabled", False),
            lean4_enabled=getattr(args, "validator.lean4_enabled", False),
            embedding_model=getattr(args, "validator.embedding_model", "all-MiniLM-L6-v2"),
        )


def create_parser(neuron_type: str = "miner") -> argparse.ArgumentParser:
    """Create CLI argument parser with common + neuron-specific args."""
    parser = argparse.ArgumentParser(
        description=f"ReasonForge {neuron_type.title()} Neuron",
    )

    # Common args
    parser.add_argument("--netuid", type=int, required=True, help="Subnet UID")
    parser.add_argument(
        "--subtensor.network",
        type=str,
        default="finney",
        help="Bittensor network (finney|test|local)",
    )
    parser.add_argument("--subtensor.chain_endpoint", type=str, default=None)
    parser.add_argument("--logging.debug", action="store_true")
    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")

    # Add bittensor native args if available
    if HAS_BITTENSOR:
        bt.Wallet.add_args(parser)
        bt.Subtensor.add_args(parser)
        bt.logging.add_args(parser)

    # Add neuron-specific args
    if neuron_type == "miner":
        MinerConfig.add_args(parser)
    elif neuron_type == "validator":
        ValidatorConfig.add_args(parser)

    return parser

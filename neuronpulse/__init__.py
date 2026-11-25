"""NeuronPulse: Trainium2 UltraServer Observability, Cross-Node Error Correlator & NKI Inference Profiler."""

from .copilot import ask_cluster_copilot
from .correlator import CrossNodeErrorCorrelator, execute_automated_repair
from .profiler import compare_serving_configs, simulate_neuron_serving
from .topology import FAULT_CATALOG, UltraServerCluster

__version__ = "1.0.0"

__all__ = [
    "CrossNodeErrorCorrelator",
    "FAULT_CATALOG",
    "UltraServerCluster",
    "ask_cluster_copilot",
    "compare_serving_configs",
    "execute_automated_repair",
    "simulate_neuron_serving",
]

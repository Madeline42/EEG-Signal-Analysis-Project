"""
CTET Signal Processing Tools 
This module provides utilities for loading and preprocessing EEG data
following the methodology of O'Connell et al. (2009)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from obci_readmanager.signal_processing.read_manager import ReadManager


# Signal scaling factor for OBCI amplifier
VOLTAGE_SCALING = 0.0715


# --- Data Structures (The "Headers") ---
@dataclass(frozen=True)
class EEGMetadata:
    """
    Stores immutable EEG session parameters.

    Attributes:
        fs (float): Sampling frequency in Hz.
        num_channels (int): Total number of recorded channels.
        channel_names (List[str]): List of channel labels (e.g., 'Pz', 'Fz').
        channel_map (Dict[str, int]): Dictionary mapping channel names to their matrix indices.
        tags (Any): Experimental tags/events from OBCI ReadManager.
    """
    fs: float
    num_channels: int
    channel_names: List[str]
    channel_map: Dict[str, int]
    tags: Any


def file_load(xml_file: str, raw_file: str, tag_file: str) -> ReadManager:
    """
    Initializes a ReadManager object to access OBCI data.

    Args:
        xml_file (str): Path to the XML configuration file.
        raw_file (str): Path to the raw EEG data file.
        tag_file (str): Path to the tags/events file.

    Returns:
        ReadManager: An initialized object for reading samples and tags.
    """
    return ReadManager(xml_file, raw_file, tag_file)


# --- Extraction Functions ---
def get_session_metadata(read_manager: ReadManager) -> EEGMetadata:
    """
    Parses ReadManager to fill the EEGMetadata structure.

    Args:
        read_manager (ReadManager): The manager object connected to EEG files.

    Returns:
        EEGMetadata: Immutable data structure containing session header.
    """
    params = read_manager.get_params()
    ch_names = params.get("channel_names", [])

    return EEGMetadata(
        fs=float(params.get("sampling_frequency", 512.0)),
        num_channels=int(params.get("number_of_channels", 0)),
        channel_names=ch_names,
        channel_map={name: i for i, name in enumerate(ch_names)},
        tags=read_manager.get_tags()
    )


def get_eeg_signal(read_manager: ReadManager) -> np.ndarray:
    """
    Extracts and scales the raw signal.
    Separate from metadata for better modularity.

    Args:
        read_manager (ReadManager): The manager object connected to EEG files.

    Returns:
        np.ndarray: Scaled EEG signal matrix in microvolts [channels x samples].

    Raises:
        ValueError: If the ReadManager returns no data.
    """
    signal = read_manager.get_samples()
    if signal is None or signal.size == 0:
        raise ValueError("readManager returned empty signal data.")

    # CRITICAL: Scaling is multiplication, not additional!
    signal *= VOLTAGE_SCALING
    return signal

"""
CTET Signal Processing Tools 
This module provides utilities for loading and preprocessing EEG data
following the methodology of O'Connell et al. (2009)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any
from scipy import signal as sp_signal
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


def file_load(xml_file: str,
              raw_file: str,
              tag_file: str
              ) -> ReadManager:
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


def apply_bandpass_filter(
        data: np.ndarray,
        fs: float,
        lowcut: float,
        highcut: float,
        order: int = 4
) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter using Second-Order Section (SOS).

    Thsi method is numerically stable for high-order filters and low frequencies 
    compared to the standard 'ba' (Transfer Funtion) representation.

    Args:
       data (np.ndarray): The EEG signal matrix [channels x samples].
       fs (float): Sampling frequency in Hz.
       lowcut (float): Lower bound of the frequency band.
       highcut (float): High bound of the frequency band.
       order (int): Yhe order of the filter. Default 4.

    Returns:
       np.ndarray: The zero-phase filtered signal.
    """
    # Defensive axis validation.
    if data.shape[0] > data.shape[1]:
        print("Warning: Input data appears to be transposed. Correcting...")
        data = data.T

    # Design the filter in SOS format.
    sos = sp_signal.butter(
        order,
        [lowcut, highcut],
        btype='bandpass',
        output='sos',
        fs=fs
    )

    # Apply the filter forward and backward (zero-phase shift).
    # This is equivalent to filtfilt but for SOS.
    filtered_data = sp_signal.sosfiltfilt(sos, data, axis=-1)

    return filtered_data


def apply_notch_filter(
        data: np.ndarray,
        fs: float,
        freq: float = 50,
        quality_factor: float = 30
) -> np.ndarray:
    """
    Applies an IIR Notch filter to remove specific power line noise.

    The filter is converted a Second-Order Section (SOS) for numerical stability.

    Args:
        data (np.ndarray): The EEG signal matrix [channels x samples].
        fs (float): Sampling frequency in Hz.
        freq (float): The frequency to be removed (e.g., 50Hz for EU power lines).
        quality_factor (float): The Q-factor, determining the notch width. Default is 30.

    Returns:
        np.ndarray: The signal with the specific frequency removed. 
    """
    # Defensive axis validation.
    if data.shape[0] > data.shape[1]:
        print("Warning: Input data appears to be transposed. Correcting...")
        data = data.T

    # Design Notch filter (Transfer Function representation).
    b, a = sp_signal.iirnotch(freq, quality_factor, fs)

    # Convert to SOS for stability (important for high precision).
    sos = sp_signal. tf2sos(b, a)

    # Apply zero-phase filtration.
    filtered_data = sp_signal.sosfiltfilt(sos, data, axis=-1)

    return filtered_data


def reference(
        data: np.ndarray,
        metadata: EEGMetadata,
        ref_channels: list = None
) -> np.ndarray:
    """
    Applies re-referencing to the EEG signal.

    Args:
        data (np.ndarray): The EEG signal matrix [channels x samples].
        metadata (EEGMetadata): Array with names and indices of channels.
        ref_channels (List): List of channels names for reference. If empty, apply Common Average Reference.

    Returns:
        np.ndarray: The signal after reference apply.
    """
    # Create a deep copy to avoid modyfing original data.
    data_copy = data.copy()  # Added parentheses!

    # 2. Handle Comon Average Reference.
    if not ref_channels:
        # slice(None) is the internal Python equivalent of ":".
        # It selects ALL indices efficiently.
        ref_indices = slice(None)
        print("Status: Applied Common Average Reference (CAR).")

    # Handle Specific Channel Reference.
    else:
        try:
            ref_indices = [metadata.channel_map[ch] for ch in ref_channels]
            print(f"Status: Applied reference to channels: {ref_channels}")
        except KeyError as e:
            # If a channel is missing, we stop and return the original copy.
            print(
                f"Critical Error: Channel {e} not found in metadata. Skipping reference.")
            return data_copy

    # The ONLY math call - univeral for both CAR and specific channels.
    # If ref_indices is slice(None), it takes all rows.
    # If ref_indices is a list, it takes only those rows.
    ref_data = np.mean(data_copy[ref_indices, :], axis=0)

    # Substract reference signal from all channels.
    data_copy -= ref_data

    return data_copy

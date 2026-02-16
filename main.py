from src.utils.ctet_tools import file_load, get_session_metadata, get_eeg_signal, apply_bandpass_filter, apply_notch_filter, reference


def main():
    # 1. Define paths (Adjust these to your local files!)
    xml_p = "data/D_01/D_01_procedura_1.obci.xml"
    raw_p = "data/D_01/D_01_procedura_1.obci.raw"
    tag_p = "data/D_01/D_01_procedura_1.tag"

    print("---Starting Analysis ---")

    try:
        # 2. Load data
        manager = file_load(xml_p, raw_p, tag_p)

        # 3. Get Metadata
        metadata = get_session_metadata(manager)
        print(
            f"Status: Successfully loaded: {metadata.num_channels} channels at {metadata.fs}Hz")

        # 4. Get signal
        raw_signal = get_eeg_signal(manager)
        print(
            f"Status: Initial signal shape: {raw_signal.shape} (Channels x Samples)")

        # 5. Re-referencing
        # Option A: Applying Common Average Reference (CAR) by default - pass empty list or None
        print("Astatus: Applying CAR reference...")
        ref_signal = reference(data=raw_signal, metadata=metadata)

        # Option B: Specific Reference
        # print("Status: Applying Linked Mastoids Reference...")
        # ref_signaf = reference(data=raw_signal, metadata=metadata, ref_channels=["A1", "A2"])

        # 6. Filtering Phase
        # Step A: Remove 50Hz power line noise (and optionally 100Hz harmonic)
        print("Status: Applying Notch filter (50Hz)...")
        notch_signal = apply_notch_filter(
            data=ref_signal, fs=metadata.fs, freq=50.0)

        # Step B: Broadband filter for 1-40 Hz
        print("Status: Applying Bandpass filter (1-40 Hz)...")
        preprocessed_signal = apply_bandpass_filter(data=notch_signal,
                                                    fs=metadata.fs,
                                                    lowcut=1.0,
                                                    highcut=40
                                                    )
        # 7. Final Verification
        print(f"--- Preprocessing Complete ---")
        print(f"Final signal shape: {preprocessed_signal.shape}")

    except Exception as e:
        print(f"Critical error during execution: {e}")


# This is the "Pythonic" way to start a program
if __name__ == "__main__":
    main()

from src.utils.ctet_tools import file_load, get_session_metadata, get_eeg_signal

def main():
    # 1. Define paths (Adjust these to your local files!)
    xml_p = "data\D_01\D_01_procedura_1.obci.xml"
    raw_p = "data\D_01\D_01_procedura_1.obci.raw"
    tag_p = "data\D_01\D_01_procedura_1.tag"

    print("---Starting Analysis ---")

    try:
        # 2. Load data
        manager = file_load(xml_p, raw_p, tag_p)

        # 3. Get Metadata
        metadata = get_session_metadata(manager)
        print(f"Succesfully loaded: {metadata.num_channels} channels at {metadata.fs}Hz")

        # 4. Get signal
        signal = get_eeg_signal(manager)
        print(f"Signal shape: {signal.shape} (Channels x Samples)")
        print("Everything works perfectly!")


    except Exception as e:
        print(f"Error during execution: {e}")

#This is the "Pythonic" way to start a program
if __name__ == "__main__":
    main()
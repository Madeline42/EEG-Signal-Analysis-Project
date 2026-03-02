# CTET EEG Signal Processing Pipeline

This repository contains a Python-based pipeline for preprocessing and analyzing EEG signals. The data analysis and code development are conducted independently, while the raw EEG dataset was collected collaboratively by a 3-person team as part of a university course. This codebase represents the initial analytical phase of a larger project intended for an upcoming scientific article.

## Project Overview
This project aims to replicate the analytical methodology of the study by [O'Connell et al. (2009)](https://doi.org/10.1523/JNEUROSCI.5967-08.2009) regarding the electrocortical signals preceding lapses of sustained attention. 

While the data processing pipeline follows their established methodology, the analysis is performed on a novel EEG dataset collected by our 3-person team. The experimental procedure, the Continuous Temporal Expectancy Task (CTET), was administered using PsychoPy. The task required participants to continuously monitor a stream of patterned stimuli and detect a rare target stimulus defined by its longer duration.

*(Note: Detailed information regarding the EEG hardware setup, specific participant demographics, and the full experimental methodology will be added in future updates as the research progresses).*

## Current Status
At this stage, the repository contains a fully functional preprocessing pipeline tailored for OpenBCI data formats.

**Working Features:**
* **Data Loading & Metadata:** Parsing OpenBCI files (`.xml`, `.raw`, `.tag`) and applying the necessary voltage scaling to the raw signal.
* **Signal Filtering (Zero-phase SOS):** * A bandpass filter (1-40 Hz) to isolate relevant brainwave frequencies.
  * A 50 Hz Notch filter for power line noise removal.
* **Re-referencing:** Common Average Reference (CAR) applied by default, with built-in support for specific channel referencing.
* **Event Tag Cleaning (`clean_raw_tags`):** A sanitization function that filters out `none` tags and rapid duplicate event markers (using a <0.1s threshold).

**Work in Progress:**
* **Task Logic Parsing (`parse_ctet_logic`):** The script logic required to fully parse CTET events (e.g., distinguishing between standard and target trials, and calculating inter-target intervals) is currently incomplete and under active development.

## Quick Start
1. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt

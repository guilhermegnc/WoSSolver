# Words on Stream - Anagram Solver

An advanced anagram solver for the "Words on Stream" game, featuring a graphical user interface (GUI) and a Convolutional Neural Network (CNN) for automatic letter detection from screenshots. This tool helps players find the best possible words from a given set of letters, including support for wildcards and identifying potentially incorrect letters.

## Features

- **Graphical User Interface (GUI):** An intuitive interface built with Tkinter for ease of use.
- **CNN-Powered Letter Detection:** Automatically detects letters from an image (e.g., a screenshot of the game) using a trained Convolutional Neural Network.
- **Anagram Solver:** Finds all possible words from a given set of letters using an efficient Trie-based algorithm.
- **Wildcard Support:** Includes support for a '?' wildcard character, testing all possible letters and providing probabilistic results.
- **False Letter Detection:** Helps identify a potentially incorrect letter in the set by analyzing the impact of its removal.
- **Image Processing:** In-app tool to crop and process images for letter detection.
- **Clipboard Integration:** Easily paste images from the clipboard (Ctrl+V) for quick analysis.
- **Results Management:** Copy and save the generated word lists.
- **Internationalization (i18n):** The user interface and dictionaries are available in multiple languages (e.g., English and Portuguese), with settings managed in a `settings.json` file.

## Requirements

- Python 3.8+
- Pillow
- NumPy
- TensorFlow
- OpenCV-Python

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Create and activate a virtual environment:**
   ```sh
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```sh
   # For the main application
   pip install -r requirements.txt

   # For model training (optional)
   pip install -r requirements-training.txt
   ```

## Usage

1. **Run the application:**
   - **On Windows:** Double-click the `WordsOnStream.bat` file.
   - **On other systems (or from the command line):**
     ```sh
     python src/solver.py
     ```

2. **Using the application:**
   - **To detect letters from an image:**
     - Copy an image of the game to your clipboard.
     - Press the "Paste Image (Ctrl+V)" button or use the Ctrl+V shortcut.
     - A dialog will appear, allowing you to crop the area with the letters.
     - The detected letters will appear in the input field.
   - **To manually enter letters:**
     - Type the letters directly into the input field.
     - Use a `?` for any wildcard or unrecognized letter.
   - **To find words:**
     - Click the "Solve" button. The results will be displayed in the text area below, grouped by word length.

## Internationalization

The application supports multiple languages for both the user interface and the dictionaries.

- **UI Translations:** The UI strings are stored in `data/lang/ui_strings.json`. You can edit this file to add new languages or modify existing translations.
- **Dictionaries:** Dictionaries are located in the `data/dictionaries/` directory. The application will load the dictionary corresponding to the language selected in the UI.
- **Settings:** Your language preferences are saved in a `settings.json` file in the root directory.

## Model Training

The CNN model for letter detection can be trained using the `src/training/train.py` script.

1. **Prepare the dataset:**
   - Create a `data/dataset` directory in the root of the project.
   - Inside the `dataset` directory, place individual images of each letter you want to train on.
   - The images should be named after the letter they represent (e.g., `A.png`, `B.png`, `C.png`).
   - Include an image for the wildcard/hidden letter, named `HIDDEN.png`.

2. **Run the training script:**
   ```sh
   python src/training/train.py
   ```
   The script will automatically load the images, generate augmented data to improve model robustness, and train the CNN.

3. **Output:**
   - The trained model will be saved to the `model/` directory (e.g., `letter_detector_final.h5`).
   - The class mappings will be saved in `model/label_map.json`.

## File Descriptions

- **`src/solver.py`:** The main entry point for the application.
- **`src/ui.py`:** Contains the main `WordsStreamGUI` class, which defines the entire graphical user interface using Tkinter.
- **`src/anagram_solver.py`:** Implements the core anagram-solving logic, including wildcard and false-letter detection, using a `Trie` data structure.
- **`src/letter_detector.py`:** A wrapper for the TensorFlow/Keras CNN model used to predict letters from images.
- **`src/segment.py`:** A module with functions for segmenting and isolating letter tiles from an image.
- **`src/trie.py`:** Implements the Trie data structure for efficient dictionary lookups.
- **`src/training/train.py`:** The script for training the letter detection CNN model.
- **`model/letter_detector_final.h5`:** The pre-trained CNN model for letter detection.
- **`model/label_map.json`:** A JSON file mapping the model's output classes to letters.
- **`data/dictionaries/`:** This directory contains the dictionaries used by the anagram solver (e.g., `pt_BR.txt`).
- **`data/lang/ui_strings.json`:** A JSON file containing the UI translations for supported languages (e.g., English and Portuguese).
- **`WordsOnStream.bat`:** A batch script for launching the application on Windows.

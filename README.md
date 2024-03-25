# Spectra-Sense ReadMe

## Setup

1. **Python Installation:** Ensure that Python version 3.7.16 is installed on your system.

2. **Git Installation:** Install Git on your system to clone the repository.

3. **Clone Git Repository:**
   - Use the provided repository link to clone the Git repository onto your local machine.

4. **Create a Virtual Environment:**
   - Run the following commands to create a virtual environment:
     ```
     python -m venv <env_name>
     source <env_name>/bin/activate
     ```
   - Activate the virtual environment to isolate your project dependencies.

5. **Upgrade pip:**
   - Once the virtual environment is activated, upgrade pip using the following command:
     ```
     python -m pip install --upgrade pip
     ```
   - Make sure the virtual environment is activated, and you are in the correct working directory (i.e., the cloned repository).

6. **Install Required Packages:**
   - Install the necessary packages listed in `requirements.txt` using pip:
     ```
     pip install -r requirements.txt
     ```

## Usage

1. **Sample Prediction Pipeline (main.py):**
   - Refer to `main.py` for a sample prediction pipeline.
   - Follow the instructions in `main.py` to make a sample prediction.

2. **Prediction Endpoint (model_endpoint.py):**
   - Utilize `model_endpoint.py` for the preprocessing pipeline and prediction endpoint.
   - Use the `make_prediction` function, providing the file path (.wav format only) as input.

3. **Output:**
   - The `make_prediction` function returns two values: the predicted emotion and probabilities for each individual emotion.

4. **Audio File Requirement:**
   - Ensure that the audio file provided for prediction is at least 5 seconds long in duration.

## Description (Optional)

- **data_loader.py:** Contains pipelines for processing training data.

- **model_loader.py:** Responsible for loading saved models from the registry and returning the model instance.

- **model_sm_md_lg.py:** Contains the architecture of the model.

- **models_sharded.py:** Contains the architecture of one of the main components of the model.

- **testing.py:** Contains a pipeline responsible for testing the model.

- **train.py:** Contains a pipeline responsible for training the model.

Follow the setup instructions to prepare your environment and utilize the provided scripts and modules for training, testing, and making predictions with your emotion prediction model.
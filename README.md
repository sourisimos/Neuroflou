This is an interpretable by design neural network. The project was a first work on this type of structure.

# Setting up the Environment and Configuration

To get started with the software, follow these steps in the terminal in the *specified order*:

bash

cd neuro-fuzzy/code
python3 -m venv my-env-name   # Create a virtual environment
source my-env-name/bin/activate
pip install --proxy=proxy:80 --upgrade pip   # Update pip
pip install --proxy=proxy:80 -r requirements.txt   # Install required libraries

The environment is now set up!
# Training the Network

To initiate network training:

    Modify parameters in INIT.py to set the desired configuration. Open INIT.py from the file manager.

    Run the following commands in the terminal:

    bash

    cd neuro-fuzzy/code
    python3 main.py (-args)   # Additional arguments can be specified to override settings directly from the terminal

Upon execution, a "Started" message will appear after a few seconds, followed by a loading screen. When the progress reaches 100%, training is complete.

After training, output files are saved in code/output_temp:

* **console_log**: Contains numerical network information such as parameters, weight values after training, and cost function values for training and validation data.
* **graph**: Shows the evolution of cost functions for the different networks, with promising networks highlighted over the total number of iterations (including non-promising neighbor networks).
* **membership_func**: Contains the membership functions for various linguistic descriptors.

# Adding a New Raw Dataset

To add a new dataset:

    Create a new folder with the desired name under neuro-fuzzy/datasets.

    Save the raw data file in this newly created folder.

    Open a terminal and follow these commands in the specified order:

    bash

    cd neuro-fuzzy/code
    source my-env-name/bin/activate
    python3 data_creator.py -h
    python3 data_creator.py -args   # Specify mandatory arguments here!

    Note: You must manually add a new entry to the netw_variables dictionary in INIT.py for network variables, using the key format <FOLDER_NAME>_<SAMPLE_NAME>.

# Adding a New Version of an Existing Dataset

If updating an existing dataset:

bash

python3 data_creator.py -args   # Not all score columns are mandatory!

    Note: As with new datasets, add a new entry to the netw_variables dictionary in INIT.py with network variables, using the key format <FOLDER_NAME>_<SAMPLE_NAME>.

# Additional Resources

Refer to the "Directory Structure" PDF, which outlines the code structure and the relationships between files, focusing on the three main files:

    main.py
    NeuroFuzzyNetwork.py
    TrainingTree.py

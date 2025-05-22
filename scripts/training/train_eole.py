import os
import sys
import logging
from eole.bin.main import main as eole_main

"""

Basic training for the eole system. Follows eole docs in passing configuration parameters.

"""



def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Eole training arguments
    args = [
        'train',
        '--config', 'configs/eole/training.yaml',
        '--model', 'configs/eole/model_vit.yaml', 
        '--data', 'configs/eole/data.yaml'
    ]
    
    # Run Eole training
    sys.argv = ['eole'] + args
    eole_main()

if __name__ == "__main__":
    main()
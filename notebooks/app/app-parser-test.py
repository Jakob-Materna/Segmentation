import yaml
import sys
from pathlib import Path

MODELS = {
  "vit_b": "sam_vit_b_01ec64.pth",
  "vit_l": "sam_vit_l_0b3195.pth",
  "vit_h": "sam_vit_h_4b8939.pth"
}

def parse_config(file_path):
    try:
        # Resolve the absolute path of the config file
        config_path = Path(file_path).resolve()
        if not config_path.is_file():
            print(f"Error: Config file not found at {config_path}.")
            exit()
        
        # Load the config file
        with config_path.open('r') as file:
            data = yaml.safe_load(file)
        
        # Resolve the absolute path for the checkpoint, if present
        checkpoint_path = Path(data["checkpoint"]).resolve()
        if checkpoint_path.is_file():
            data["checkpoint"] = str(checkpoint_path)
        else:
            print(f"Error: Checkpoint file not found at: {checkpoint_path}")
            exit()

        print("Config file parsed successfully.")
        return data

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    file_path = "config.yaml"  
    config = parse_config(file_path)
    if config:
        print(config)
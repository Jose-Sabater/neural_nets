import yaml
from dataclasses import dataclass
import activation_functions as af
import loss

# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")


@dataclass
class conf:
    """
    Reads from the config.yaml file and stores the values for our network
    """

    @staticmethod
    def choose_function(activation: str) -> tuple[callable, callable]:
        # Map the functions
        func_eval = {
            "tanh": (af.tanh, af.tanh_prime),
            "softmax": (af.softmax, af.softmax_prime),
            "mse": (loss.mse, loss.mse_prime),
        }
        return func_eval[activation]

    network_activation: tuple = choose_function(config["network_activation"])
    output_activation: tuple = choose_function(config["output_activation"])
    loss_functions: tuple = choose_function(config["loss_function"])
    epochs: int = config["epochs"]
    learning_rate = config["learning_rate"]

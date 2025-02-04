import tensorflow as tf
import visualkeras

def load_models_from_dict(model_paths):
    """
    Load models from a dictionary of file paths.
    Args:
        model_paths (dict): Dictionary where keys are model names and values are paths to model files.

    Returns:
        dict: Dictionary of loaded models where keys are model names.
    """
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = tf.keras.models.load_model(path, compile=False)
            print(f"Loaded model: {name}")
        except Exception as e:
            print(f"Failed to load model {name} from {path}: {e}")
    return models


def draw_model_architectures(models, save_dir=None, legend_fontsize=50):
    """
    Draw and save model architectures using visualkeras.
    Args:
        models (dict): Dictionary of loaded models where keys are model names.
        save_dir (str, optional): Directory to save the diagrams. If None, only displays the diagrams.
        legend_fontsize (int, optional): Font size for the legend. Defaults to 12.
    """
    for name, model in models.items():
        print(f"Drawing architecture for model: {name}")
        if save_dir:
            filepath = f"{save_dir}/{name}_architecture.png"
            visualkeras.layered_view(
                model, 
                to_file=filepath, 
                legend={'fontsize': legend_fontsize},
                draw_volume=False  # Disable volume rendering
            ).show()
            print(f"Saved architecture diagram to {filepath}")
        else:
            visualkeras.layered_view(
                model, 
                legend={'fontsize': legend_fontsize},
                draw_volume=False  # Disable volume rendering
            ).show()


# Example usage:
    
MPATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/postprocs/sel_models"
model_files = {
    'p07a_n01_l01_i01_b01_r01':	f'{MPATH}/p07a_n01_l01_i01_b01_r01_model.keras',
    # 'p07a_n01_l01_i02_b01_r01':	f'{MPATH}/p07a_n01_l01_i02_b01_r01_model.keras',
    # 'p07a_n01_l01_i03_b01_r01':	f'{MPATH}/p07a_n01_l01_i03_b01_r01_model.keras',
    # 'p07a_n01_l02_i01_b01_r01':	f'{MPATH}/p07a_n01_l02_i01_b01_r01_model.keras',
    # 'p07a_n01_l02_i02_b01_r01':	f'{MPATH}/p07a_n01_l02_i02_b01_r01_model.keras',
    # 'p07a_n01_l02_i03_b01_r01':	f'{MPATH}/p07a_n01_l02_i03_b01_r01_model.keras',
    'p07a_n02_l01_i01_b01_r01':	f'{MPATH}/p07a_n02_l01_i01_b01_r01_model.keras',
    # 'p07a_n02_l01_i02_b01_r01':	f'{MPATH}/p07a_n02_l01_i02_b01_r01_model.keras',
    # 'p07a_n02_l01_i03_b01_r01':	f'{MPATH}/p07a_n02_l01_i03_b01_r01_model.keras',
    # 'p07a_n02_l02_i01_b01_r01':	f'{MPATH}/p07a_n02_l02_i01_b01_r01_model.keras',
    # 'p07a_n02_l02_i02_b01_r01':	f'{MPATH}/p07a_n02_l02_i02_b01_r01_model.keras',
    # 'p07a_n02_l02_i03_b01_r01':	f'{MPATH}/p07a_n02_l02_i03_b01_r01_model.keras',
    'p07a_n03_l01_i01_b01_r01':	f'{MPATH}/p07a_n03_l01_i01_b01_r01_model.keras',
    # 'p07a_n03_l01_i02_b01_r01':	f'{MPATH}/p07a_n03_l01_i02_b01_r01_model.keras',
    # 'p07a_n03_l01_i03_b01_r01':	f'{MPATH}/p07a_n03_l01_i03_b01_r01_model.keras',
    # 'p07a_n03_l02_i01_b01_r01':	f'{MPATH}/p07a_n03_l02_i01_b01_r01_model.keras',
    # 'p07a_n03_l02_i02_b01_r01':	f'{MPATH}/p07a_n03_l02_i02_b01_r01_model.keras',
    # 'p07a_n03_l02_i03_b01_r01':	f'{MPATH}/p07a_n03_l02_i03_b01_r01_model.keras',
    'p07a_s01_l01_i01_b01_r01':	f'{MPATH}/p07a_s01_l01_i01_b01_r01_model.keras',
    # 'p07a_s01_l01_i02_b01_r01':	f'{MPATH}/p07a_s01_l01_i02_b01_r01_model.keras',
    # 'p07a_s01_l01_i03_b01_r01':	f'{MPATH}/p07a_s01_l01_i03_b01_r01_model.keras',
    # 'p07a_s01_l02_i01_b01_r01':	f'{MPATH}/p07a_s01_l02_i01_b01_r01_model.keras',
    # 'p07a_s01_l02_i02_b01_r01':	f'{MPATH}/p07a_s01_l02_i02_b01_r01_model.keras',
    # 'p07a_s01_l02_i03_b01_r01':	f'{MPATH}/p07a_s01_l02_i03_b01_r01_model.keras',
    'p07a_s02_l01_i01_b01_r01':	f'{MPATH}/p07a_s02_l01_i01_b01_r01_model.keras',
    # 'p07a_s02_l01_i02_b01_r01':	f'{MPATH}/p07a_s02_l01_i02_b01_r01_model.keras',
    # 'p07a_s02_l01_i03_b01_r01':	f'{MPATH}/p07a_s02_l01_i03_b01_r01_model.keras',
    # 'p07a_s02_l02_i01_b01_r01':	f'{MPATH}/p07a_s02_l02_i01_b01_r01_model.keras',
    # 'p07a_s02_l02_i02_b01_r01':	f'{MPATH}/p07a_s02_l02_i02_b01_r01_model.keras',
    # 'p07a_s02_l02_i03_b01_r01':	f'{MPATH}/p07a_s02_l02_i03_b01_r01_model.keras',
    'p07a_s03_l01_i01_b01_r01':	f'{MPATH}/p07a_s03_l01_i01_b01_r01_model.keras',
    # 'p07a_s03_l01_i02_b01_r01':	f'{MPATH}/p07a_s03_l01_i02_b01_r01_model.keras',
    # 'p07a_s03_l01_i03_b01_r01':	f'{MPATH}/p07a_s03_l01_i03_b01_r01_model.keras',
    # 'p07a_s03_l02_i01_b01_r01':	f'{MPATH}/p07a_s03_l02_i01_b01_r01_model.keras',
    # 'p07a_s03_l02_i02_b01_r01':	f'{MPATH}/p07a_s03_l02_i02_b01_r01_model.keras',
    # 'p07a_s03_l02_i03_b01_r01':	f'{MPATH}/p07a_s03_l02_i03_b01_r01_model.keras',
    'p07a_u01_l01_i01_b01_r01':	f'{MPATH}/p07a_u01_l01_i01_b01_r01_model.keras',
    # 'p07a_u01_l01_i02_b01_r01':	f'{MPATH}/p07a_u01_l01_i02_b01_r01_model.keras',
    # 'p07a_u01_l01_i03_b01_r01':	f'{MPATH}/p07a_u01_l01_i03_b01_r01_model.keras',
    # 'p07a_u01_l02_i01_b01_r01':	f'{MPATH}/p07a_u01_l02_i01_b01_r01_model.keras',
    # 'p07a_u01_l02_i02_b01_r01':	f'{MPATH}/p07a_u01_l02_i02_b01_r01_model.keras',
    # 'p07a_u01_l02_i03_b01_r01':	f'{MPATH}/p07a_u01_l02_i03_b01_r01_model.keras',
    'p07a_u02_l01_i01_b01_r01':	f'{MPATH}/p07a_u02_l01_i01_b01_r01_model.keras',
    # 'p07a_u02_l01_i02_b01_r01':	f'{MPATH}/p07a_u02_l01_i02_b01_r01_model.keras',
    # 'p07a_u02_l01_i03_b01_r01':	f'{MPATH}/p07a_u02_l01_i03_b01_r01_model.keras',
    # 'p07a_u02_l02_i01_b01_r01':	f'{MPATH}/p07a_u02_l02_i01_b01_r01_model.keras',
    # 'p07a_u02_l02_i02_b01_r01':	f'{MPATH}/p07a_u02_l02_i02_b01_r01_model.keras',
    # 'p07a_u02_l02_i03_b01_r01':	f'{MPATH}/p07a_u02_l02_i03_b01_r01_model.keras',
}

save_directory = MPATH

# Load models
loaded_models = load_models_from_dict(model_files)

# Draw and save model architectures
draw_model_architectures(loaded_models, save_dir=save_directory)

#%%

import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

path = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/postprocs/sel_models/p07a_u01_l01_i01_b01_r01_model.keras"
m = tf.keras.models.load_model(path, compile=False)
m.summary()
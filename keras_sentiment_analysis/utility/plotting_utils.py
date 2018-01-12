from keras.utils.vis_utils import plot_model


def plot_and_save_model(model, filename):
    plot_model(model, show_shapes=True, to_file=filename)

import tensorflow as tf
import keras.backend as K


def export_keras_to_tensorflow(keras_model, output_fld, output_model_file,
                               output_graphdef_file=None,
                               num_output=None,
                               quantize=False,
                               save_output_graphdef_file=False,
                               output_node_prefix=None):
    K.set_learning_phase(0)

    if output_graphdef_file is None:
        output_graphdef_file = 'model.ascii'
    if num_output is None:
        num_output = 1
    if output_node_prefix is None:
        output_node_prefix = 'output_node'

    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = output_node_prefix + str(i)
        pred[i] = tf.identity(keras_model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    if save_output_graphdef_file:
        tf.train.write_graph(sess.graph.as_graph_def(), output_fld, output_graphdef_file, as_text=True)
        print('saved the graph definition in ascii format at: ', output_graphdef_file)

    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from tensorflow.tools.graph_transforms import TransformGraph
    if quantize:
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', output_model_file)


def export_text_model_to_csv(config, output_fld, output_model_file):
    word2idx = config['word2idx']
    max_len = config['max_len']
    labels = config['labels']

    file_path = output_fld + '/' + output_model_file
    with open(file_path, 'wt', encoding='utf-8') as f:
        f.write(str(max_len) + '\n')
        for label, index in labels.items():
            f.write('label\t' + label + '\t' + str(index) + '\n')
        for word, index in word2idx.items():
            f.write(word + '\t' + str(index) + '\n')


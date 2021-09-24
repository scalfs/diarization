import numpy as np

import uisrnn

SAVED_MODEL_NAME = 'voxcon_dev_model.uisrnn'

def diarization_experiment(model_args, training_args, inference_args):

    train_sequence = np.load('/app/fixed-voxcon-dev-sequences.npy', allow_pickle=True)
    train_cluster_id = np.load('/app/voxsrc21-dia/embeddings/sequences/voxcon-dev-cluster-ids.npy', allow_pickle=True)
    
    concatenated_train_sequence = np.concatenate(train_sequence)
    concatenated_train_cluster_id = np.concatenate(train_cluster_id)
    
    model = uisrnn.UISRNN(model_args)
    # Training.
    # If we have saved a mode previously, we can also skip training by
    # calling：
    # model.load(SAVED_MODEL_NAME)
    model.fit(concatenated_train_sequence, concatenated_train_cluster_id, training_args)
    model.save(SAVED_MODEL_NAME)
    
    
def main():
    """The main function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    print(training_args)
    diarization_experiment(model_args, training_args, inference_args)


if __name__ == "__main__":
    main()
    print('Program completed!')
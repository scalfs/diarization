import numpy as np

import uisrnn

SAVED_MODEL_NAME = 'voxcon_dev_model.uisrnn'

def diarization_experiment(model_args, training_args, inference_args):

    train_sequence = np.load('/app/fixed-voxcon-dev-sequences.npy', allow_pickle=True)
    train_cluster_id = np.load('/app/voxsrc21-dia/embeddings/sequences/voxcon-dev-cluster-ids.npy', allow_pickle=True)
    
    # Training.
    # If we have saved a mode previously, we can also skip training by
    # calling：
    # model.load(SAVED_MODEL_NAME)
    model.fit(train_sequence, train_cluster_id, training_args)
    model.save(SAVED_MODEL_NAME)
    
    
    
def main():
    """The main function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    print(
    diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
    main()
import numpy as np

import uisrnn

SAVED_MODEL_NAME = 'voxcon_dev_model.uisrnn'

def diarization_experiment(model_args, training_args, inference_args):

    train_sequence = np.load('/app/fixed-voxcon-dev-sequences.npy', allow_pickle=True)
    train_cluster_id = np.load('/app/voxsrc21-dia/embeddings/sequences/voxcon-dev-cluster-ids.npy', allow_pickle=True)
    
    # How many elements each 
    # list should have 
    n = 53

    # using list comprehension 
    split_train_sequence = [train_sequence[i:i + n] for i in range(0, len(train_sequence), n)]
    split_train_cluster_id = [train_cluster_id[i:i + n] for i in range(0, len(train_cluster_id), n)]
    
    training_args.train_iteration = 210
    model = uisrnn.UISRNN(model_args)
    
    for sequence, cluster_id in zip(split_train_sequence, split_train_cluster_id):
        concatenated_train_sequence = np.concatenate(sequence)
        concatenated_train_cluster_id = np.concatenate(cluster_id)
    
        # Training
        model.fit(concatenated_train_sequence, concatenated_train_cluster_id, training_args)
        print('foi')
        model.save(SAVED_MODEL_NAME)
    
    
    
    
def main():
    """The main function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    print(training_args)
    diarization_experiment(model_args, training_args, inference_args)


if __name__ == "__main__":
    main()
    print('Program completed!')
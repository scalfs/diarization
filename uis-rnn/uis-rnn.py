import numpy as np
from functools import partial
import torch.multiprocessing as mp
ctx = mp.get_context('forkserver')

import uisrnn

SAVED_MODEL_NAME = 'voxcon_dev_model.uisrnn'

NUM_WORKERS = 2

def diarization_experiment(model_args, training_args, inference_args):

    train_sequence = np.load('/app/fixed-voxcon-dev-sequences.npy', allow_pickle=True).tolist()
    train_cluster_id = np.load('/app/voxsrc21-dia/embeddings/sequences/voxcon-dev-cluster-ids.npy', allow_pickle=True).tolist()
    
    test_sequences = np.load('/app/fixed-voxcon-test-sequences.npy', allow_pickle=True).tolist()
    test_cluster_ids = np.load('/app/voxsrc21-dia/embeddings/sequences/voxcon-test-cluster-ids.npy', allow_pickle=True).tolist()
    
    # How many elements each list should have
    n = 53

    # using list comprehension 
    split_train_sequence = [train_sequence[i:i + n] for i in range(0, len(train_sequence), n)]
    split_train_cluster_id = [train_cluster_id[i:i + n] for i in range(0, len(train_cluster_id), n)]
    
    training_args.train_iteration = 300
    model = uisrnn.UISRNN(model_args)
    
    for sequence, cluster_id in zip(split_train_sequence, split_train_cluster_id):
        # concatenated_train_sequence = np.concatenate(sequence)
        # concatenated_train_cluster_id = np.concatenate(cluster_id)
    
        # Training
        model.fit(sequence, cluster_id, training_args)        
        model.save(SAVED_MODEL_NAME)

    
    # testing
    predicted_cluster_ids = []
    test_record = []
    # predict sequences in parallel
    model.rnn_model.share_memory()
    pool = ctx.Pool(NUM_WORKERS, maxtasksperchild=None)
    pred_gen = pool.imap(func=partial(model.predict, args=inference_args), iterable=test_sequences)
    # collect and score predicitons
    for idx, predicted_cluster_id in enumerate(pred_gen):
        accuracy = uisrnn.compute_sequence_match_accuracy(test_cluster_ids[idx], predicted_cluster_id)
        predicted_cluster_ids.append(predicted_cluster_id)
        test_record.append((accuracy, len(test_cluster_ids[idx])))
        print('Ground truth labels:')
        print(test_cluster_ids[idx])
        print('Predicted labels:')
        print(predicted_cluster_id)
        print('-' * 80)

    # close multiprocessing pool
    pool.close()

    print('Finished diarization experiment')
    print(uisrnn.output_result(model_args, training_args, test_record))
    
    
    
    
def main():
    """The main function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    print(model_args, training_args, inference_args)
    diarization_experiment(model_args, training_args, inference_args)


if __name__ == "__main__":
    main()
    print('Program completed!')
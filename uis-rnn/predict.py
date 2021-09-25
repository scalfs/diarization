from pyannote.core import Annotation, Segment
import uisrnn
import json
import numpy as np
from functools import partial
import torch.multiprocessing as mp
ctx = mp.get_context('forkserver')

# sequences of segment-level embeddings
voxsrc21_seqs = np.load(
    '/app/sequences/voxsrc21-sequences.npy', allow_pickle=True).tolist()

# segments intervals, for creating resulting RTTM and calculating DER and JER
voxsrc21_intervals = np.load(
    '/app/sequences/voxsrc21-intervals.npy', allow_pickle=True).tolist()

# processing order of audios, saved in sequence
voxsrc21_audio_ids = ["tpine", "huqpy", "uzsvq", "pirso", "npnyp", "jsatj", "kuunx", "klbrr", "ecugo", "dwtow", "lhbqr", "zfipy", "wsobx", "jxkah", "hjvtj", "dpbgj", "phcxl", "uddju", "hufrz", "fibgq", "ahrcr", "cnbhz", "ahpuj", "biypq", "taukp", "iwvbh", "zobzz", "chvgo", "zsngo", "ixbvn", "ruwbp", "dufvn", "hajxp", "ohjjf", "veada", "zfzwt", "aqlrr", "qvroz", "cqfbd", "hzttx", "jrxnz", "gcccl", "qgzvk", "ieaoq", "jwezt", "ovxrk", "oitjh", "xivtm", "rnmrp", "euhwq", "iflmj", "hffvn", "eoxkf", "ycycy", "fwjhb", "yhntc", "ersxa", "afhlm", "uhpjz", "saenk", "oszsp", "zkmgn", "cgjuc", "vofje", "ajjmr", "mkbjy", "cxrjf", "jbirg", "hmnyo", "mknge", "xahab", "svnoe", "iucho", "uwgqx", "sfjjj", "rymcy", "rycwr", "fdtlj", "ewkei", "pojyu", "vrgwd", "bfwgq", "goots", "mrbme", "nzoid", "cehwp", "uosun", "pzrcf", "kpecv", "pwutv", "ipszk", "ktmre", "bylqb", "mrgru", "osrss", "ftajg", "xairp", "whibh", "xmkio", "rvyug", "tealt", "ezptn", "kpihx", "ubthy", "njuto", "komgz", "ahcuo", "lcowh", "vuqbp", "wnoyw", "zliuj", "dvofo", "yhivq", "xvaqn", "ixcig", "zauxp", "nzonc", "rvrlo", "xesmw", "mylgb", "askiw", "nsibk", "quinj", "swpjb", "fsfdw", "kcoeu", "eeivl", "tuczd", "jjbsx", "ytvbn", "dwsdu",
                      "fseqv", "zjrun", "pqifp", "aqind", "enrur", "vtdos", "ymvrw", "welcq", "wazie", "ssbra", "kxzac", "tcwva", "epbjz", "hkfjg", "bwpth", "cnwba", "vghwm", "usxym", "yyldd", "govov", "ueoss", "ivmhq", "gsigu", "sotzc", "ebixn", "zhcic", "ckivy", "tldnh", "hiukp", "lvrse", "sdpur", "ubitk", "reljk", "hfzhn", "axabh", "bmldz", "ujdjd", "kgzrb", "msnqr", "syucd", "ognux", "ahorw", "ifafa", "bpbon", "yywxf", "qezbm", "sljce", "nttgg", "uupjc", "gukfd", "hexbr", "jnoor", "rfyhi", "pbocz", "eqalx", "tpfau", "aoehz", "hmkzx", "zdymj", "qxezs", "irswh", "zktli", "dkpkx", "cqhil", "kxjyn", "rjpos", "kqlln", "vqzkm", "jucfh", "wjwes", "mebni", "ubogn", "obsbu", "dsuiy", "fmtir", "alpmg", "sndtu", "tkxsz", "qnuxe", "ctjzs", "uelsx", "xwaec", "bzlts", "fftup", "fqpdh", "ffcio", "wofgd", "oiyjs", "avvvz", "tbedi", "zyjfv", "bgkus", "linoq", "ptctn", "fhjbj", "wmhps", "lceba", "msswz", "rxldb", "lqvix", "vwgph", "zsgyz", "anojt", "lwxbp", "slldp", "jpjcy", "xpewn", "bamsl", "qmsgw", "wayap", "eizev", "nbema", "sqtvn", "oewsu", "rjdar", "rjmds", "vwftp", "ckeyp", "fcghr", "jfkiy", "ogkch", "iadsf", "jrnnn", "vjkrd", "brxew", "uyykg", "xikii", "zskbo", "xmezt", "kygkd", "rpjuz", "bmria", "eumxr"]

# batches of 264/4 (parallel cpus) sequences
n = 66
i = 0
test_seqs = voxsrc21_seqs[i*n:(i+1)*n]
test_intervals = voxsrc21_intervals[i*n:(i+1)*n]
test_audio_ids = voxsrc21_audio_ids[i*n:(i+1)*n]

SAVED_MODEL_NAME = 'voxcon_full_model.uisrnn'
NUM_WORKERS = 3


def predict(model_args, inference_args):

    model_args.crp_alpha = 0.76
    model = uisrnn.UISRNN(model_args)

    model.load(SAVED_MODEL_NAME)

    # testing
    predicted_cluster_ids = {}

    # predict sequences in parallel
    model.rnn_model.share_memory()
    pool = ctx.Pool(NUM_WORKERS, maxtasksperchild=None)
    pred_gen = pool.imap(func=partial(
        model.predict, args=inference_args), iterable=test_seqs)
    # collect and score predicitons
    for idx, predicted_cluster_id in enumerate(pred_gen):
        audio_id = test_audio_ids[idx]
        predicted_cluster_ids[audio_id] = predicted_cluster_id

        print('Predicted labels:',
              audio_id, f'{idx}/{n}', f'batch{i}')
        print(predicted_cluster_id)
        print('-' * 80)

        annotation = Annotation()
        for jdx, speaker_id in enumerate(predicted_cluster_id):
            segment_interval = test_intervals[idx][jdx]
            annotation[Segment(segment_interval[0],
                               segment_interval[1])] = speaker_id

        rttmFile = '/app/rttm/{}.rttm'.format(audio_id)
        with open(rttmFile, 'w') as writeRttmFile:
            annotation.support(0.401).write_rttm(writeRttmFile)

    # open file for writing, "w"
    predicted_path = f'/app/predict-{i}.json'
    f = open(predicted_path, "w")
    predicted_ids = json.dumps(predicted_cluster_ids)
    f.write(predicted_ids)
    f.close()

    # close multiprocessing pool
    pool.close()

    print('Finished diarization experiment')


def main():
    """The main function."""
    model_args, _, inference_args = uisrnn.parse_arguments()
    print(model_args, inference_args)
    predict(model_args, inference_args)


if __name__ == "__main__":
    main()
    print('Program completed!')

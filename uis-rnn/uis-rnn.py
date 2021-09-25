import os
import re
import contrib
import uisrnn
import numpy as np
from functools import partial
import torch.multiprocessing as mp
ctx = mp.get_context('forkserver')


base_dir = '/app'
# rttm_path = f'{base_dir}/voxsrc21-dia/data/voxconverse/sample/rttm'
rttm_path = os.path.join(base_dir, 'datasets/voxconverse/test/rttm')
sequences_path = os.path.join(base_dir, 'voxsrc21-dia/embeddings/sequences')
os.makedirs(sequences_path, exist_ok=True)

# speaker assignment should be processed with the audios, but since they weren't
# these are from the logs, to replicate the files order on embedding extraction (and interval generation)
dev_audio_ids = ["abjxc", "ahnss", "akthc", "asxwr", "atgpi", "aufkn", "azisu", "bkwns", "bspxd", "bxpwa", "cyyxp", "czlvt", "dhorc", "djqif", "dscgs", "edixl", "ehpau", "eqttu", "esrit", "exymw", "eziem", "ezsgk", "femmv", "fkvvo", "fsaal", "fvyvb", "fxgvy", "ggvel", "gocbm", "gqbvk", "gqdxy", "grzbb", "hiyis", "hqyok", "hycgx", "ikgcq", "imtug", "ipqqq", "iqbww", "iqtde", "irvat", "jnivh", "jynhe", "kctgl", "kkghn", "ktzmw", "kuduk", "ldkmv", "lfzib", "lknjp", "luvfz", "mdbod", "mekog", "mesob", "mgpok", "migzj", "mkrcv", "mpvoh", "mqxsf", "mwfmq", "nctdh", "nfqjx", "ntchr", "nxgad", "oekmc", "ooxnm", "pgkde", "plbbw", "pnyir", "ppgjx", "qfdpp", "qouur", "qppll", "qrzjk", "qydmg", "qygfk", "qzwxa", "rcxzg", "rxgun", "sikkm", "sosnj", "suuxu", "syiwe", "szsyz", "tguxv", "tjkfn", "tlprc", "tplwz", "udjij", "uexjc", "ulriv", "uvnmy", "vmbga", "whmpa", "wjhgf", "xiglo", "xxwgv", "ycxxe", "ydlfw", "ypwjd", "yrsve", "yuzyu", "ywcwr", "zcdsd", "zfkap",
                 "zmndm", "zrlyl", "zyffh", "nnqfq", "aisvi", "usbgm", "xvllq", "oenox", "praxo", "onpra", "kefgo", "bauzd", "mjgil", "blwmj", "gofnj", "uatlu", "rtvuw", "wnfoi", "evtyi", "tcwsn", "pilgb", "cmfyw", "dbugl", "mevkw", "jsdmu", "jiqvr", "hkzpa", "hgeec", "jcako", "epdpg", "cqaec", "kkwkn", "spzmn", "ngyrk", "sldwj", "cmhsm", "ndkwv", "kbkon", "bdopb", "qhesr", "cwryz", "djngn", "dvngl", "qsfzo", "gwtwd", "cobal", "sduml", "vysqj", "jtagk", "hgdez", "wdjyj", "qpylu", "tfvyr", "falxo", "bydui", "willh", "wspbh", "ufpel", "kiadt", "nrogz", "imbqf", "crixb", "ylnza", "wewoz", "qvtia", "kszpd", "bwzyf", "xypdm", "mvjuk", "jhdav", "pqmho", "jsmbi", "ccokr", "ampme", "odkzj", "tiams", "tucrg", "bravd", "houcx", "gpjne", "goyli", "txcok", "jyirt", "oxxwk", "iwdjy", "kckqn", "ioasm", "paibn", "kklpv", "vbjlx", "jyflp", "sqkup", "xmfzh", "afjiv", "eapdk", "pnook", "yfcmz", "gzvkx", "oklol", "qjgpl", "wbqza", "wmori", "ysgbf", "zajzs", "zidwg", "ztzzr", "zvmyn"]
test_audio_ids = ["aepyx", "aiqwk", "bjruf", "bmsyn", "bxcfq", "byapz", "clfcg", "cqfmj", "crylr", "cvofp", "dgvwu", "dohag", "dxbbt", "dzsef", "eauve", "eazeq", "eguui", "epygx", "eqsta", "euqef", "fijfi", "fpfvy", "fqrnu", "fxnwf", "fyqoe", "gcfwp", "gtjow", "gtnjb", "gukoa", "guvqf", "gylzn", "gyomp", "hcyak", "heolf", "hhepf", "ibrnm", "ifwki", "iiprr", "ikhje", "jdrwl", "jjkrt", "jjvkx", "jrfaz", "jsbdo", "jttar", "jxpom", "jzkzt", "kajfh", "kmunk", "kpjud", "ktvto", "kvkje", "lbfnx", "ledhe", "lilfy", "ljpes", "lkikz", "lpola", "lscfc", "ltgmz", "lubpm", "luobn", "mjmgr", "msbyq", "mupzb", "myjoe", "nlvdr", "nprxc", "ocfop", "ofbxh", "olzkb", "ooxlj", "oqwpd", "otmpf", "ouvtt", "poucc", "ppexo", "pwnsw", "qadia", "qeejz", "qlrry", "qwepo", "rarij", "rmvsh", "rxulz", "sebyw", "sexgc", "sfdvy", "svxzm", "tkybe", "tpslg", "uedkc", "uqxlg", "usqam", "vncid", "vylyk", "vzuru", "wdvva", "wemos", "wprog", "wwzsk", "xggbk", "xkgos", "xlyov", "xmyyy", "xqxkt", "xtdcl", "xtzoq", "xvxwv", "ybhwz", "ylzez", "ytmef", "yukhy", "yzvon", "zedtj",
                  "zfzlc", "zowse", "zqidv", "zztbo", "ralnu", "uicid", "laoyl", "jxydp", "pzxit", "upshw", "gfneh", "kzmyi", "nkqzr", "kgjaa", "dkabn", "eucfa", "erslt", "mclsr", "fzwtp", "dzxut", "pkwrt", "gmmwm", "leneg", "sxqvt", "pgtkk", "fuzfh", "vtzqw", "rsypp", "qxana", "optsn", "dxokr", "ptses", "isxwc", "gzhwb", "mhwyr", "duvox", "ezxso", "jgiyq", "rpkso", "kmjvh", "wcxfk", "gcvrb", "eddje", "pccww", "vuewy", "tvtoe", "oubab", "jwggf", "aggyz", "bidnq", "neiye", "mkhie", "iowob", "jbowg", "gwloo", "uevxo", "nitgx", "eoyaz", "qoarn", "mxdpo", "auzru", "diysk", "cwbvu", "jeymh", "iacod", "cawnd", "vgaez", "bgvvt", "tiido", "aorju", "qajyo", "ryken", "iabca", "tkhgs", "tbjqx", "mqtep", "fowhl", "fvhrk", "nqcpi", "mbzht", "uhfrw", "utial", "cpebh", "tnjoh", "jsymf", "vgevv", "mxduo", "gkiki", "bvyvm", "hqhrb", "isrps", "nqyqm", "dlast", "pxqme", "bpzsc", "vdlvr", "lhuly", "crorm", "bvqnu", "tpnyf", "thnuq", "swbnm", "cadba", "sbrmv", "wibky", "wlfsf", "wwvcs", "xffsa", "xkmqx", "xlsme", "ygrip", "ylgug", "ytula", "zehzu", "zsgto", "zzsba", "zzyyo"]

# segments intervals, for creating resulting RTTM and calculating DER and JER
# voxcon_dev_intervals = np.load(f'{base_dir}/sequences/voxcon-dev-intervals.npy', allow_pickle=True)
# voxcon_test_intervals = np.load(f'{base_dir}/sequences/voxcon-test-intervals.npy', allow_pickle=True)
# voxsrc21_intervals = np.load(f'{base_dir}/sequences/voxsrc21-intervals.npy', allow_pickle=True)

# train/test_sequences -> list of sequences: M (utterances) * L (segments) * D (embeddings dimension)
# train/test_cluster_ids -> list of sequences speakers ids: M * L * 1 (string)
voxcon_dev_seqs = np.load(os.path.join(
    base_dir, 'sequences/fixed-voxcon-dev-sequences.npy'), allow_pickle=True).tolist()
voxcon_dev_ids = np.load(os.path.join(
    base_dir, sequences_path, 'voxcon-dev-cluster-ids.npy'), allow_pickle=True).tolist()

voxcon_test_seqs = np.load(os.path.join(
    base_dir, 'sequences/fixed-voxcon-test-sequences.npy'), allow_pickle=True).tolist()
voxcon_test_ids = np.load(os.path.join(
    base_dir, sequences_path, 'voxcon-test-cluster-ids.npy'), allow_pickle=True).tolist()

# voxsrc21_seqs = np.load(f'{base_dir}/sequences/voxsrc21-sequences.npy', allow_pickle=True).tolist()

# Mix voxcon dev and test sets for training model, to test on voxsrc21
smaller_len = len(voxcon_dev_ids) if len(voxcon_dev_ids) < len(
    voxcon_test_ids) else len(voxcon_test_ids)

dev_seqs = voxcon_dev_seqs[:smaller_len]
dev_remain_seqs = voxcon_dev_seqs[smaller_len:]
test_seqs = voxcon_test_seqs[:smaller_len]
test_remain_seqs = voxcon_test_seqs[smaller_len:]

mixed_seqs = np.concatenate(list(zip(dev_seqs, test_seqs))).tolist()
mixed_seqs.extend(dev_remain_seqs)
mixed_seqs.extend(test_remain_seqs)

dev_ids = voxcon_dev_ids[:smaller_len]
dev_remain_ids = voxcon_dev_ids[smaller_len:]
test_ids = voxcon_test_ids[:smaller_len]
test_remain_ids = voxcon_test_ids[smaller_len:]

mixed_ids = np.concatenate(list(zip(dev_ids, test_ids))).tolist()
mixed_ids.extend(dev_remain_ids)
mixed_ids.extend(test_remain_ids)

# CRIAR FOLDS AQUI
# intervals = dev_intervals
audio_ids = dev_audio_ids
train_sequences = mixed_seqs
train_cluster_ids = mixed_ids
test_sequences = []
test_cluster_ids = []


def calculate_crp_alpha():
    for m, segments_ids in enumerate(train_cluster_ids):
        for l, speaker_id in enumerate(segments_ids):
            # int to remove 0 in 00, 01, ...
            train_cluster_ids[m][l] = str(
                int(re.sub('[^0-9]', "", speaker_id)))

    concat_seq, concat_ids = uisrnn.utils.concatenate_training_data(
        train_sequences, train_cluster_ids)
    contrib.estimate_crp_alpha(concat_ids, search_step=0.1)


SAVED_MODEL_NAME = 'voxcon_full_model.uisrnn'
NUM_WORKERS = 2


def diarization_experiment(model_args, training_args, inference_args):
    # How many elements each list should have
    # n = 53

    # using list comprehension
    # split_train_sequence = [train_sequence[i:i + n] for i in range(0, len(train_sequence), n)]
    # split_train_cluster_id = [train_cluster_id[i:i + n] for i in range(0, len(train_cluster_id), n)]

    training_args.train_iteration = 1000
    model_args.crp_alpha = 0.76
    model = uisrnn.UISRNN(model_args)

    for i in range(3):
        model.fit(train_sequences, train_cluster_ids, training_args)
        model.save(SAVED_MODEL_NAME)

    # testing
    predicted_cluster_ids = []
    test_record = []
    # predict sequences in parallel
    model.rnn_model.share_memory()
    pool = ctx.Pool(NUM_WORKERS, maxtasksperchild=None)
    pred_gen = pool.imap(func=partial(
        model.predict, args=inference_args), iterable=test_sequences[:5])
    # collect and score predicitons
    for idx, predicted_cluster_id in enumerate(pred_gen):
        accuracy = uisrnn.compute_sequence_match_accuracy(
            test_cluster_ids[idx], predicted_cluster_id)
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
    # model_args, training_args, inference_args = uisrnn.parse_arguments()
    # print(model_args, training_args, inference_args)
    # diarization_experiment(model_args, training_args, inference_args)
    calculate_crp_alpha()


if __name__ == "__main__":
    main()
    print('Program completed!')

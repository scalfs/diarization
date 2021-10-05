import numpy as np
from pyannote.core import Annotation, Segment
from implementation import spectral_clustering

base_dir = '/home/jovyan/work'
rttm_dir = '{}/voxsrc21-spectral-v2-rttm'.format(base_dir)

# sequences of segment-level embeddings
# voxcon_test_seqs = np.load(
#     f'{base_dir}/sequences/fixed-voxcon-test-sequences.npy', allow_pickle=True).tolist()
voxsrc21_seqs = np.load(
    f'{base_dir}/sequences/voxsrc21-sequences.npy', allow_pickle=True).tolist()

# # segments intervals, for creating resulting RTTM and calculating DER and JER
# voxcon_test_intervals = np.load(
#     f'{base_dir}/sequences/fixed-voxcon-test-intervals.npy', allow_pickle=True).tolist()
voxsrc21_intervals = np.load(
    f'{base_dir}/sequences/voxsrc21-intervals.npy', allow_pickle=True).tolist()

# ids sorted by processing order (regarding to seqs and interval clusters)
voxcon_test_audio_ids = ["aepyx", "aiqwk", "bjruf", "bmsyn", "bxcfq", "byapz", "clfcg", "cqfmj", "crylr", "cvofp", "dgvwu", "dohag", "dxbbt", "dzsef", "eauve", "eazeq", "eguui", "epygx", "eqsta", "euqef", "fijfi", "fpfvy", "fqrnu", "fxnwf", "fyqoe", "gcfwp", "gtjow", "gtnjb", "gukoa", "guvqf", "gylzn", "gyomp", "hcyak", "heolf", "hhepf", "ibrnm", "ifwki", "iiprr", "ikhje", "jdrwl", "jjkrt", "jjvkx", "jrfaz", "jsbdo", "jttar", "jxpom", "jzkzt", "kajfh", "kmunk", "kpjud", "ktvto", "kvkje", "lbfnx", "ledhe", "lilfy", "ljpes", "lkikz", "lpola", "lscfc", "ltgmz", "lubpm", "luobn", "mjmgr", "msbyq", "mupzb", "myjoe", "nlvdr", "nprxc", "ocfop", "ofbxh", "olzkb", "ooxlj", "oqwpd", "otmpf", "ouvtt", "poucc", "ppexo", "pwnsw", "qadia", "qeejz", "qlrry", "qwepo", "rarij", "rmvsh", "rxulz", "sebyw", "sexgc", "sfdvy", "svxzm", "tkybe", "tpslg", "uedkc", "uqxlg", "usqam", "vncid", "vylyk", "vzuru", "wdvva", "wemos", "wprog", "wwzsk", "xggbk", "xkgos", "xlyov", "xmyyy", "xqxkt", "xtdcl", "xtzoq", "xvxwv", "ybhwz", "ylzez", "ytmef", "yukhy", "yzvon", "zedtj",
                         "zfzlc", "zowse", "zqidv", "zztbo", "ralnu", "uicid", "laoyl", "jxydp", "pzxit", "upshw", "gfneh", "kzmyi", "nkqzr", "kgjaa", "dkabn", "eucfa", "erslt", "mclsr", "fzwtp", "dzxut", "pkwrt", "gmmwm", "leneg", "sxqvt", "pgtkk", "fuzfh", "vtzqw", "rsypp", "qxana", "optsn", "dxokr", "ptses", "isxwc", "gzhwb", "mhwyr", "duvox", "ezxso", "jgiyq", "rpkso", "kmjvh", "wcxfk", "gcvrb", "eddje", "pccww", "vuewy", "tvtoe", "oubab", "jwggf", "aggyz", "bidnq", "neiye", "mkhie", "iowob", "jbowg", "gwloo", "uevxo", "nitgx", "eoyaz", "qoarn", "mxdpo", "auzru", "diysk", "cwbvu", "jeymh", "iacod", "cawnd", "vgaez", "bgvvt", "tiido", "aorju", "qajyo", "ryken", "iabca", "tkhgs", "tbjqx", "mqtep", "fowhl", "fvhrk", "nqcpi", "mbzht", "uhfrw", "utial", "cpebh", "tnjoh", "jsymf", "vgevv", "mxduo", "gkiki", "bvyvm", "hqhrb", "isrps", "nqyqm", "dlast", "pxqme", "bpzsc", "vdlvr", "lhuly", "crorm", "bvqnu", "tpnyf", "thnuq", "swbnm", "cadba", "sbrmv", "wibky", "wlfsf", "wwvcs", "xffsa", "xkmqx", "xlsme", "ygrip", "ylgug", "ytula", "zehzu", "zsgto", "zzsba", "zzyyo"]

voxsrc21_audio_ids = ["tpine", "huqpy", "uzsvq", "pirso", "npnyp", "jsatj", "kuunx", "klbrr", "ecugo", "dwtow", "lhbqr", "zfipy", "wsobx", "jxkah", "hjvtj", "dpbgj", "phcxl", "uddju", "hufrz", "fibgq", "ahrcr", "cnbhz", "ahpuj", "biypq", "taukp", "iwvbh", "zobzz", "chvgo", "zsngo", "ixbvn", "ruwbp", "dufvn", "hajxp", "ohjjf", "veada", "zfzwt", "aqlrr", "qvroz", "cqfbd", "hzttx", "jrxnz", "gcccl", "qgzvk", "ieaoq", "jwezt", "ovxrk", "oitjh", "xivtm", "rnmrp", "euhwq", "iflmj", "hffvn", "eoxkf", "ycycy", "fwjhb", "yhntc", "ersxa", "afhlm", "uhpjz", "saenk", "oszsp", "zkmgn", "cgjuc", "vofje", "ajjmr", "mkbjy", "cxrjf", "jbirg", "hmnyo", "mknge", "xahab", "svnoe", "iucho", "uwgqx", "sfjjj", "rymcy", "rycwr", "fdtlj", "ewkei", "pojyu", "vrgwd", "bfwgq", "goots", "mrbme", "nzoid", "cehwp", "uosun", "pzrcf", "kpecv", "pwutv", "ipszk", "ktmre", "bylqb", "mrgru", "osrss", "ftajg", "xairp", "whibh", "xmkio", "rvyug", "tealt", "ezptn", "kpihx", "ubthy", "njuto", "komgz", "ahcuo", "lcowh", "vuqbp", "wnoyw", "zliuj", "dvofo", "yhivq", "xvaqn", "ixcig", "zauxp", "nzonc", "rvrlo", "xesmw", "mylgb", "askiw", "nsibk", "quinj", "swpjb", "fsfdw", "kcoeu", "eeivl", "tuczd", "jjbsx", "ytvbn", "dwsdu",
                      "fseqv", "zjrun", "pqifp", "aqind", "enrur", "vtdos", "ymvrw", "welcq", "wazie", "ssbra", "kxzac", "tcwva", "epbjz", "hkfjg", "bwpth", "cnwba", "vghwm", "usxym", "yyldd", "govov", "ueoss", "ivmhq", "gsigu", "sotzc", "ebixn", "zhcic", "ckivy", "tldnh", "hiukp", "lvrse", "sdpur", "ubitk", "reljk", "hfzhn", "axabh", "bmldz", "ujdjd", "kgzrb", "msnqr", "syucd", "ognux", "ahorw", "ifafa", "bpbon", "yywxf", "qezbm", "sljce", "nttgg", "uupjc", "gukfd", "hexbr", "jnoor", "rfyhi", "pbocz", "eqalx", "tpfau", "aoehz", "hmkzx", "zdymj", "qxezs", "irswh", "zktli", "dkpkx", "cqhil", "kxjyn", "rjpos", "kqlln", "vqzkm", "jucfh", "wjwes", "mebni", "ubogn", "obsbu", "dsuiy", "fmtir", "alpmg", "sndtu", "tkxsz", "qnuxe", "ctjzs", "uelsx", "xwaec", "bzlts", "fftup", "fqpdh", "ffcio", "wofgd", "oiyjs", "avvvz", "tbedi", "zyjfv", "bgkus", "linoq", "ptctn", "fhjbj", "wmhps", "lceba", "msswz", "rxldb", "lqvix", "vwgph", "zsgyz", "anojt", "lwxbp", "slldp", "jpjcy", "xpewn", "bamsl", "qmsgw", "wayap", "eizev", "nbema", "sqtvn", "oewsu", "rjdar", "rjmds", "vwftp", "ckeyp", "fcghr", "jfkiy", "ogkch", "iadsf", "jrnnn", "vjkrd", "brxew", "uyykg", "xikii", "zskbo", "xmezt", "kygkd", "rpjuz", "bmria", "eumxr"]

# To be used on prediction
sequences = voxsrc21_seqs
intervals = voxsrc21_intervals
sample_ids = voxsrc21_audio_ids

for idx, sample_id in enumerate(sample_ids):
    labels = spectral_clustering(sequences[idx])

    print('Predicted labels: ', sample_id, f' {idx+1}/{len(sample_ids)}')

    annotation = Annotation()
    annotation.uri = sample_id
    for jdx, speaker_id in enumerate(labels):
        segment_interval = intervals[idx][jdx]
        annotation[Segment(segment_interval[0],
                           segment_interval[1])] = speaker_id

    rttm_file = '{}/{}.rttm'.format(rttm_dir, sample_id)
    with open(rttm_file, 'w') as file:
        annotation.support().write_rttm(file)

    # rttm_file_collar = '{}/rttm_colar/{}.rttm'.format(rttm_dir, sample_id)
    # with open(rttm_file_collar, 'w') as file:
    #     annotation.support(0.481).write_rttm(file)

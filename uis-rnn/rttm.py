def xor(x, y):
    """Return truth value of ``x`` XOR ``y``."""
    return bool(x) != bool(y)


def format_float(x, n_digits=3):
    """Format floating point number for output as string.
    Parameters
    ----------
    x : float
        Number.
    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)
    Returns
    -------
    s : str
        Formatted string.
    """
    fmt_str = '%%.%df' % n_digits
    return fmt_str % round(x, n_digits)


class Turn(object):
    """Speaker turn class.
    A turn represents a segment of audio attributed to a single speaker.
    Parameters
    ----------
    onset : float
        Onset of turn in seconds from beginning of recording.
    offset : float, optional
        Offset of turn in seconds from beginning of recording. If None, then
        computed from ``onset`` and ``dur``.
        (Default: None)
    dur : float, optional
        Duration of turn in seconds. If None, then computed from ``onset`` and
        ``offset``.
        (Default: None)
    speaker_id : str, optional
        Speaker id.
        (Default: None)
    file_id : str, optional
        File id.
        (Default: none)
    """

    def __init__(self, onset, offset=None, dur=None, speaker_id=None,
                 file_id=None):
        if not xor(offset is None, dur is None):
            raise ValueError('Exactly one of offset or dur must be given')
        if onset < 0:
            raise ValueError('Turn onset must be >= 0 seconds')
        if offset:
            dur = offset - onset
        if dur <= 0:
            raise ValueError('Turn duration must be > 0 seconds')
        if not offset:
            offset = onset + dur
        self.onset = onset
        self.offset = offset
        self.dur = dur
        self.speaker_id = speaker_id
        self.file_id = file_id

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.onset, self.offset, self.dur, self.file_id,
                     self.speaker_id))

    def __str__(self):
        return ('FILE: %s, SPEAKER: %s, ONSET: %f, OFFSET: %f, DUR: %f' %
                (self.file_id, self.speaker_id, self.onset, self.offset,
                 self.dur))

    def __repr__(self):
        speaker_id = ("'%s'" % self.speaker_id if self.speaker_id is not None
                      else None)
        file_id = ("'%s'" % self.file_id if self.file_id is not None
                   else None)
        return ('Turn(%f, %f, None, %s, %s)' %
                (self.onset, self.offset, speaker_id, file_id))


def _parse_rttm_line(line):
    line = line.decode('utf-8').strip()
    fields = line.split()
    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)
    file_id = fields[1]
    speaker_id = fields[7]

    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError:
        raise IOError('Turn onset not FLOAT. LINE: "%s"' % line)
    if onset < 0:
        raise IOError('Turn onset < 0 seconds. LINE: "%s"' % line)

    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError:
        raise IOError('Turn duration not FLOAT. LINE: "%s"' % line)
    if dur <= 0:
        raise IOError('Turn duration <= 0 seconds. LINE: "%s"' % line)

    return Turn(onset, dur=dur, speaker_id=speaker_id, file_id=file_id)


def load_rttm(rttmf):
    """Load speaker turns from RTTM file.
    For a description of the RTTM format, consult Appendix A of the NIST RT-09
    evaluation plan.
    Parameters
    ----------
    rttmf : str
        Path to RTTM file.
    Returns
    -------
    turns : list of Turn
        Speaker turns.
    speaker_ids : set
        Speaker ids present in ``rttmf``.
    file_ids : set
        File ids present in ``rttmf``.
    References
    ----------
    NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition
    Evaluation Plan. https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    with open(rttmf, 'rb') as f:
        turns = []
        speaker_ids = set()
        file_ids = set()
        for line in f:
            if line.startswith(b'SPKR-INFO'):
                continue
            turn = _parse_rttm_line(line)
            turns.append(turn)
            speaker_ids.add(turn.speaker_id)
            file_ids.add(turn.file_id)
    return turns, speaker_ids, file_ids


def write_rttm(rttmf, turns, n_digits=3):
    """Write speaker turns to RTTM file.
    For a description of the RTTM format, consult Appendix A of the NIST RT-09
    evaluation plan.
    Parameters
    ----------
    rttmf : str
        Path to output RTTM file.
    turns : list of Turn
        Speaker turns.
    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)
    References
    ----------
    NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition
    Evaluation Plan. https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    with open(rttmf, 'wb') as f:
        for turn in turns:
            fields = ['SPEAKER',
                      turn.file_id,
                      '1',
                      format_float(turn.onset, n_digits),
                      format_float(turn.dur, n_digits),
                      '<NA>',
                      '<NA>',
                      turn.speaker_id,
                      '<NA>',
                      '<NA>']
            line = ' '.join(fields)
            f.write(line.encode('utf-8'))
            f.write(b'\n')

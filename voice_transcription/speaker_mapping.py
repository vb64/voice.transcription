"""Speaker mapping."""
import nltk

from . import add_log

SENTENCE_END = ".?!"


def get_word_ts_anchor(s, e, option="start"):
    """Get word timestamp anchor."""
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    """Get first word index of sentence."""
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in SENTENCE_END
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    """Get last word index of sentence."""
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in SENTENCE_END
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    """Make sentences."""
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_realigned_ws_mapping_with_punctuation(
  word_speaker_mapping,
  max_words_in_sentence=50
):
    """Make punctuation."""
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in SENTENCE_END
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx: right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx: right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    """Make words mapping."""
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping


def speaker_timestamps(rttm_file):
    """Make speakers timestamps from Nemo rttm file."""
    speaker_ts = []
    with open(rttm_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    return speaker_ts


def map_speakers(call_log, rttm_file, whisper_word_timestamps):
    """Merge speakers data from Whisper and Nemo."""
    start_time = add_log(call_log, "map_speakers", None)

    speaker_ts = speaker_timestamps(rttm_file)
    start_time = add_log(call_log, "speaker_timestamps", start_time)

    wsm = get_words_speaker_mapping(whisper_word_timestamps, speaker_ts, "start")
    start_time = add_log(call_log, "get_words_speaker_mapping", start_time)

    with_punctuation = get_realigned_ws_mapping_with_punctuation(wsm)
    start_time = add_log(call_log, "get_realigned_ws_mapping_with_punctuation", start_time)

    ssm = get_sentences_speaker_mapping(
      with_punctuation,
      speaker_ts
    )
    add_log(call_log, "get_sentences_speaker_mapping", start_time)

    return ssm

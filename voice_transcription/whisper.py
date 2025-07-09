"""Whisper segments processing."""


def msec(sec):
    """Return int milliseconds for numpy float seconds."""
    return int(round(float(sec * 1000)))


def segments_to_json(segments, _total_msec):
    """Decode whisper segments to json."""
    data = []
    for segment in segments:
        seg_data = [msec(segment.start), msec(segment.end - segment.start), segment.text.strip()]
        words = []
        for word in segment.words:
            words.append([msec(word.start), msec(word.end - word.start), word.word.strip()])
        seg_data.append(words)
        data.append(seg_data)

    return data

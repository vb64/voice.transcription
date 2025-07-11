"""Whisper segments processing."""


def msec(sec):
    """Return int milliseconds for numpy float seconds."""
    return int(round(float(sec * 1000)))


def segments_to_json(segments, total_msec, progress_bar):
    """Decode whisper segments to json."""
    progress_bar(0, total_msec)

    data = []
    first_segment_offset = None

    for segment in segments:
        seg_data = [msec(segment.start), msec(segment.end - segment.start), segment.text.strip()]
        if first_segment_offset is None:
            first_segment_offset = seg_data[0]

        progress_bar(seg_data[0] - first_segment_offset, total_msec)

        words = []
        for word in segment.words:
            words.append([msec(word.start), msec(word.end - word.start), word.word.strip()])
        seg_data.append(words)
        data.append(seg_data)

    return data

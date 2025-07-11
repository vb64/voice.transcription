"""Whisper segments processing."""
from datetime import datetime
import faster_whisper


def msec(sec):
    """Return int milliseconds for numpy float seconds."""
    return int(round(float(sec * 1000)))


def segments_to_json(segments, total_msec, progress_bar, utc_time_start):
    """Decode whisper segments to json."""
    progress_bar(0, total_msec, utc_time_start)

    data = []
    first_segment_offset = None

    for segment in segments:
        seg_data = [msec(segment.start), msec(segment.end - segment.start), segment.text.strip()]
        if first_segment_offset is None:
            first_segment_offset = seg_data[0]

        progress_bar(seg_data[0] - first_segment_offset, total_msec, utc_time_start)

        words = []
        for word in segment.words:
            words.append([msec(word.start), msec(word.end - word.start), word.word.strip()])
        seg_data.append(words)
        data.append(seg_data)

    return data


def make_json(whisper_model, mp3_file, progress_bar):
    """Return json data for given mp3 file name."""
    print("Decode {} to wave ...".format(mp3_file))
    waveform = faster_whisper.decode_audio(mp3_file)

    print("Creating segments...")
    segments, info = whisper_model.transcribe(
      waveform, 'ru', suppress_tokens=[-1],
      vad_filter=True,
      word_timestamps=True
    )
    duration = msec(info.duration_after_vad)
    print("Duration", duration, "msec")

    now = datetime.utcnow()
    data = segments_to_json(segments, duration, progress_bar, now)
    progress_bar(duration, duration, now)

    return data

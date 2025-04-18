"""Srt file."""
from . import add_log


def format_timestamp(
    milliseconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = "."
):
    """Convert timestamp to string."""
    if milliseconds < 0:
        raise ValueError("Non-negative timestamp expected: {}".format(milliseconds))

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_srt(call_log, transcript, file_name):
    """Write a transcript to a file in SRT format."""
    start_time = add_log(call_log, "write_srt", None)
    out = open(file_name, "w", encoding="utf-8")

    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(i, file=out)
        print(
          format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=','),
          "-->",
          "{}".format(format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')),
          file=out
        )
        print(
          "{}:".format(segment['speaker']),
          "{}\n".format(segment['text'].strip().replace('-->', '->')),
          file=out
        )

    out.close()
    add_log(call_log, "done write_srt", start_time)

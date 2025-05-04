import sys
import audioread


def main(file_name):
    sec = audioread.audio_open(file_name).duration
    print("#", file_name, "length", sec, "sec")


if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1])

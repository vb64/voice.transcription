SENTENCE_END = ".!?"


def diarization(in_file, out_file):
    print(in_file, "->", out_file)


def content(in_file, out_file):
    print(in_file, "->", out_file)


def is_complete(sentence):
    return sentence[-1] in SENTENCE_END


def split_to_sentences(text):
    sentences = []
    current = []
    for word in text.split():
        current.append(word)
        if word[-1] in SENTENCE_END:
            sentences.append(' '.join(current))
            current = []

    if current:
        raise Exception("Wrong sentence: {}".format(text))

    return sentences


def write_sentence(out, text):
    for i in split_to_sentences(text):
        out.write(i)
        out.write('\n')


def whisper(in_file, out_file):
    print(in_file, "->", out_file)
    lines = open(in_file, "rt", encoding="utf-8").readlines()
    start = 0
    text = []
    eof = False
    while not eof:
        chunk = lines[start:start + 4]
        start += 4
        if len(chunk) < 4:
            eof = True
        else:
            text.append(chunk[2].strip())

    out = open(out_file, "wt", encoding="utf-8")
    group = []
    for sentence in text:
        group.append(sentence)
        if is_complete(sentence):
            write_sentence(out, ' '.join(group))
            group = []

    out.close()


def main():
    whisper("whisper.srt", "whisper.txt")
    diarization("diarization.srt", "diarization.txt")
    content("content.md", "content.txt")


if __name__ == '__main__':
    main()

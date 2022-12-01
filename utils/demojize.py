import re
import emoji


def demojize_text(text):
    regex_pattern = re.compile("["
                               u"\U00000000-\U000FFFFF"

                               "]+", flags=re.UNICODE)
    temp = ''
    for a in text:
        print('a: ', a)
        if regex_pattern.match(a):
            temp += emoji.demojize(a).replace(':', ' ').replace('_', ' ')
        else:
            temp += a
    print(text)

    return text


def remove_emoji(text):
    regex_pattern = re.compile("["
                               u"\U00000000-\U000FFFFF"

                               "]+", flags=re.UNICODE)
    temp = ''
    temp1 = regex_pattern.sub(emoji.demojize(text), text)
    for a in text:
        if regex_pattern.match(a):
            temp += emoji.demojize(a).replace(':', ' ').replace('_', ' ')

        else:
            temp += a
    temp2 = regex_pattern.findall(text)
    return temp

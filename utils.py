import re


def get_label_from_filename(filename):
    pattern = 'v_([a-zA-Z]*)_'
    match = re.search(pattern, filename).groups()[0]
    return match

# if __name__ == '__main__':
# print(get_label_from_filename('v_BenchPress_g01_c01.avi'))

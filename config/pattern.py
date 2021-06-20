import re

BREAK_PATTERN = re.compile(r'\n')
EMOJI_PATTERN = re.compile(r'\\uf\w+')
SPACES_PATTERN = re.compile(r' +')
WEB_PATTERN = re.compile('[\w\-_\d]*.(com|net)', re.I)
NUM_PATTERN = re.compile(r'\d+')

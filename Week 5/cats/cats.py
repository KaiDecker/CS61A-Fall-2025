"""Typing test implementation"""

from utils import (
    lower,
    split,
    remove_punctuation,
    lines_from_file,
    count,
    deep_convert_to_tuple,
)
from ucb import main, interact, trace
from datetime import datetime
import random


###########
# Phase 1 #
###########


def pick(paragraphs: list[str], select, k: int) -> str:
    """Return the Kth paragraph from PARAGRAPHS for which the SELECT function returns True.
    If there are fewer than K such paragraphs, return an empty string.

    Arguments:
        paragraphs: a list of strings representing paragraphs
        select: a function that returns True for paragraphs that meet its criteria
        k: an integer representing which paragraph to return

    >>> ps = ['hi', 'how are you', 'fine']
    >>> s = lambda p: len(p) <= 4
    >>> pick(ps, s, 0)
    'hi'
    >>> pick(ps, s, 1)
    'fine'
    >>> pick(ps, s, 2)
    ''
    """
    # BEGIN PROBLEM 1
    count = 0 # 使用 count 来确定输出第几个段落
    for p in paragraphs:
        if select(p):
            if count == k:
                return p
            count += 1
    return ''
    # END PROBLEM 1


def about(keywords: list[str]):
    """Return a function that takes in a paragraph and returns whether
    that paragraph contains one of the words in keywords.

    Arguments:
        keywords: a list of keywords

    >>> about_dogs = about(['dog', 'dogs', 'pup', 'puppy'])
    >>> pick(['Cute Dog!', 'That is a cat.', 'Nice pup!'], about_dogs, 0)
    'Cute Dog!'
    >>> pick(['Cute Dog!', 'That is a cat.', 'Nice pup.'], about_dogs, 1)
    'Nice pup.'
    """
    assert all([lower(x) == x for x in keywords]), "keywords should be lowercase."

    # BEGIN PROBLEM 2
    def select(paragraph: str) -> bool:
        # 去除标点符号
        cleaned_paragraph = remove_punctuation(paragraph.lower())
        # 分割成单词，并放入集合
        words_in_paragraph = set(cleaned_paragraph.split())
        # 有单词的话就返回 True
        return any(keyword in words_in_paragraph for keyword in keywords)
    return select
    # END PROBLEM 2


def accuracy(typed: str, source: str) -> float:
    """Return the accuracy (percentage of words typed correctly) of TYPED
    compared to the corresponding words in SOURCE.

    Arguments:
        typed: a string that may contain typos
        source: a model string without errors

    >>> accuracy('Cute Dog!', 'Cute Dog.')
    50.0
    >>> accuracy('A Cute Dog!', 'Cute Dog.')
    0.0
    >>> accuracy('cute Dog.', 'Cute Dog.')
    50.0
    >>> accuracy('Cute Dog. I say!', 'Cute Dog.')
    50.0
    >>> accuracy('Cute', 'Cute Dog.')
    100.0
    >>> accuracy('', 'Cute Dog.')
    0.0
    >>> accuracy('', '')
    100.0
    """
    typed_words = split(typed)
    source_words = split(source)
    # BEGIN PROBLEM 3
    # 如果 typed 为空，而 source 不为空
    if not typed_words and source_words:
        return 0.0
    # 如果都为空
    if not typed_words and not source_words:
        return 100.0
    
    # 找到匹配的数量
    matches = 0
    for t, s in zip(typed_words, source_words):
        if t == s:
            matches += 1

    total_words = len(typed_words)

    return (matches / total_words) * 100
    # END PROBLEM 3


def wpm(typed: str, elapsed: int) -> float:
    """Return the words-per-minute (WPM) of the TYPED string.

    Arguments:
        typed: an entered string
        elapsed: an amount of time in seconds

    >>> wpm('hello friend hello buddy hello', 15)
    24.0
    >>> wpm('0123456789',60)
    2.0
    """
    assert elapsed > 0, "Elapsed time must be positive"
    # BEGIN PROBLEM 4
    total_characters = len(typed)
    return (total_characters / 5) / (elapsed / 60)
    # END PROBLEM 4


################
# Phase 4 (EC) #
################


def memo(f):
    """A general memoization decorator."""
    cache = {}

    def memoized(*args):
        immutable_args = deep_convert_to_tuple(args)  # convert *args into a tuple representation
        if immutable_args not in cache:
            result = f(*immutable_args)
            cache[immutable_args] = result
            return result
        return cache[immutable_args]

    return memoized


def memo_diff(diff_function):
    """A memoization function."""
    cache = {}

    def memoized(typed, source, limit):
        # BEGIN PROBLEM EC
        "*** YOUR CODE HERE ***"
        # END PROBLEM EC

    return memoized


###########
# Phase 2 #
###########


def autocorrect(typed_word: str, word_list: list[str], diff_function, limit: int) -> str:
    """Returns the element of WORD_LIST that has the smallest difference
    from TYPED_WORD based on DIFF_FUNCTION. If multiple words are tied for the smallest difference,
    return the one that appears closest to the front of WORD_LIST. If the
    lowest difference is greater than LIMIT, return TYPED_WORD instead.

    Arguments:
        typed_word: a string representing a word that may contain typos
        word_list: a list of strings representing source words
        diff_function: a function quantifying the difference between two words
        limit: a number

    >>> ten_diff = lambda w1, w2, limit: 10 # Always returns 10
    >>> autocorrect("hwllo", ["butter", "hello", "potato"], ten_diff, 20)
    'butter'
    >>> first_diff = lambda w1, w2, limit: (1 if w1[0] != w2[0] else 0) # Checks for matching first char
    >>> autocorrect("tosting", ["testing", "asking", "fasting"], first_diff, 10)
    'testing'
    """
    # BEGIN PROBLEM 5
    if typed_word in word_list:
        return typed_word
    
    # 其中 word_list 为可迭代对象，并且使用 key 函数，用于对每个元素计算一个值
    closest_word = min(word_list, key=lambda word: diff_function(typed_word, word, limit))

    # 对返回的最小差异词计算
    diff = diff_function(typed_word, closest_word, limit)
    
    # 如果最小差异大于 limit，返回 typed_word 本身
    if diff > limit:
        return typed_word
    
    # 否则，返回差异最小的单词
    return closest_word
    # END PROBLEM 5


def furry_fixes(typed: str, source: str, limit: int) -> int:
    """A diff function for autocorrect that determines how many letters
    in TYPED need to be substituted to create SOURCE, then adds the difference in
    their lengths to this value and returns the result.

    Arguments:
        typed: a starting word
        source: a string representing a desired goal word
        limit: a number representing an upper bound on the number of chars that must change

    >>> big_limit = 10
    >>> furry_fixes("nice", "rice", big_limit)    # Substitute: n -> r
    1
    >>> furry_fixes("range", "rungs", big_limit)  # Substitute: a -> u, e -> s
    2
    >>> furry_fixes("pill", "pillage", big_limit) # Don't substitute anything, length difference of 3.
    3
    >>> furry_fixes("roses", "arose", big_limit)  # Substitute: r -> a, o -> r, s -> o, e -> s, s -> e
    5
    >>> furry_fixes("rose", "hello", big_limit)   # Substitute: r->h, o->e, s->l, e->l, length difference of 1.
    5
    """
    # BEGIN PROBLEM 6
    if abs(len(typed) - len(source)) > limit:
        return limit + 1

    # 递归计算对齐部分的替换次数
    def helper(t, s, remaining_limit):
        # 如果 limit 已经用光，直接停止
        if remaining_limit < 0:
            return limit + 1  # 返回一个超过 limit 的值，用作剪枝

        # 当任一字符串为空时，不再递归（因为替换不再发生）
        if not t or not s:
            return 0

        # 如果当前字符不同，需要 1 次替换
        if t[0] != s[0]:
            return 1 + helper(t[1:], s[1:], remaining_limit - 1)
        else:
            # 如果相同，不消耗 limit
            return helper(t[1:], s[1:], remaining_limit)

    # 先递归计算对齐部分的替换次数
    aligned_cost = helper(typed, source, limit)

    # 再加上两者长度差
    total = aligned_cost + abs(len(typed) - len(source))

    return total
    # END PROBLEM 6


def minimum_mewtations(typed: str, source: str, limit: int) -> int:
    """A diff function for autocorrect that computes the edit distance from TYPED to SOURCE.
    This function takes in a string TYPED, a string SOURCE, and a number LIMIT.

    Arguments:
        typed: a starting word
        source: a string representing a desired goal word
        limit: a number representing an upper bound on the number of edits

    >>> big_limit = 10
    >>> minimum_mewtations("cats", "scat", big_limit)       # cats -> scats -> scat
    2
    >>> minimum_mewtations("purng", "purring", big_limit)   # purng -> purrng -> purring
    2
    >>> minimum_mewtations("ckiteus", "kittens", big_limit) # ckiteus -> kiteus -> kitteus -> kittens
    3
    """
    if limit < 0: # Base cases should go here, you may add more base cases as needed.
        # BEGIN
        return limit + 1
        # END
    
    if typed == source:
        # BEGIN
        return 0
        # END

    if len(typed) == 0:
        # BEGIN
        return len(source)  # 意味着全是添加
        # END

    if len(source) == 0:
        # BEGIN
        return len(typed)  # 意味着全是删除
        # END
    
    # Recursive cases should go below here
    if typed[0] == source[0]: # Feel free to remove or add additional cases
        # BEGIN
        return minimum_mewtations(typed[1:], source[1:], limit)
        # END
    else:
        # Add: 在 typed 加一个字母即相当于跳过 source[0]
        add = 1 + minimum_mewtations(typed, source[1:], limit - 1)
        # Remove: 删掉 typed[0]
        remove = 1 + minimum_mewtations(typed[1:], source, limit - 1)
        # Substitute: typed[0] 替换成 source[0]
        substitute = 1 + minimum_mewtations(typed[1:], source[1:], limit - 1)
        # BEGIN
        return min(add, remove, substitute)
        # END


# Ignore the line below
minimum_mewtations = count(minimum_mewtations)


def final_diff(typed: str, source: str, limit: int) -> int:
    """A diff function that takes in a string TYPED, a string SOURCE, and a number LIMIT.
    If you implement this function, it will be used."""
    # ---------- Helper data ----------
    # Keyboard adjacency groups (cost 0.5 instead of 1)
    neighbors = {
        'q':'w', 'w':'qe', 'e':'wr', 'r':'et', 't':'ry',
        'y':'tu', 'u':'yi', 'i':'uo', 'o':'ip', 'p':'o',

        'a':'s', 's':'adw', 'd':'sfe', 'f':'dgr', 'g':'fht',
        'h':'gjy', 'j':'huik', 'k':'jiol', 'l':'ko',

        'z':'x', 'x':'zc', 'c':'xv', 'v':'cb', 'b':'vn',
        'n':'bm', 'm':'n'
    }

    # ---------- Base cases ----------
    if limit < 0:
        return limit + 1

    if typed == source:
        return 0

    if len(typed) == 0:
        return len(source)

    if len(source) == 0:
        return len(typed)

    # ---------- Swap detection ----------
    # Detect adjacent swap: e.g., "acress" → "caress"
    if len(typed) > 1 and len(source) > 1:
        if typed[0] == source[1] and typed[1] == source[0]:
            return 1 + final_diff(typed[2:], source[2:], limit - 1)

    # ---------- If first letters match ----------
    if typed[0] == source[0]:
        return final_diff(typed[1:], source[1:], limit)

    # ---------- Costs ----------
    # Substitute cost：如果是键盘相邻，成本更低（0.5）
    if typed[0] in neighbors and source[0] in neighbors[typed[0]]:
        sub_cost = 0.5
    else:
        sub_cost = 1

    # Repeated letter removal optimization
    # If typed contains double letter (like "tt"), removing one is cheaper
    if len(typed) > 1 and typed[0] == typed[1]:
        repeat_remove_cost = 0.5
    else:
        repeat_remove_cost = 1

    # ---------- Recursive choices ----------
    add = 1 + final_diff(typed, source[1:], limit - 1)      # Add letter
    remove = repeat_remove_cost + final_diff(typed[1:], source, limit - repeat_remove_cost)
    substitute = sub_cost + final_diff(typed[1:], source[1:], limit - sub_cost)

    return min(add, remove, substitute)


FINAL_DIFF_LIMIT = 6  # REPLACE THIS WITH YOUR LIMIT


###########
# Phase 3 #
###########


def report_progress(typed: list[str], source: list[str], user_id: int, upload) -> float:
    """Upload a report of your id and progress so far to the multiplayer server.
    Returns the progress so far.

    Arguments:
        typed: a list of the words typed so far
        source: a list of the words in the typing source
        user_id: a number representing the id of the current user
        upload: a function used to upload progress to the multiplayer server

    >>> print_progress = lambda d: print('ID:', d['id'], 'Progress:', d['progress'])
    >>> # The above function displays progress in the format ID: __, Progress: __
    >>> print_progress({'id': 1, 'progress': 0.6})
    ID: 1 Progress: 0.6
    >>> typed = ['how', 'are', 'you']
    >>> source = ['how', 'are', 'you', 'doing', 'today']
    >>> report_progress(typed, source, 2, print_progress)
    ID: 2 Progress: 0.6
    0.6
    >>> report_progress(['how', 'aree'], source, 3, print_progress)
    ID: 3 Progress: 0.2
    0.2
    """
    # BEGIN PROBLEM 8
    "*** YOUR CODE HERE ***"
    # END PROBLEM 8


def time_per_word(words: list[str], timestamps_per_player: list[list[int]]) -> dict:
    """Return a dictionary {'words': words, 'times': times} where times
    is a list of lists that stores the durations it took each player to type
    each word in words.

    Arguments:
        words: a list of words, in the order they are typed.
        timestamps_per_player: A list of lists of timestamps including the time
                          each player started typing, followed by the time each
                          player finished typing each word.

    >>> p = [[75, 81, 84, 90, 92], [19, 29, 35, 36, 38]]
    >>> result = time_per_word(['collar', 'plush', 'blush', 'repute'], p)
    >>> result['words']
    ['collar', 'plush', 'blush', 'repute']
    >>> result['times']
    [[6, 3, 6, 2], [10, 6, 1, 2]]
    """
    tpp = timestamps_per_player  # A shorter name (for convenience)
    # BEGIN PROBLEM 9
    times = []  # You may remove this line
    # END PROBLEM 9
    return {'words': words, 'times': times}


def fastest_words(words_and_times: dict) -> list[list[str]]:
    """Return a list of lists indicating which words each player typed fastest.
    In case of a tie, the player with the lower index is considered to be the one who typed it the fastest.

    Arguments:
        words_and_times: a dictionary {'words': words, 'times': times} where
        words is a list of the words typed and times is a list of lists of times
        spent by each player typing each word.

    >>> p0 = [5, 1, 3]
    >>> p1 = [4, 1, 6]
    >>> fastest_words({'words': ['Just', 'have', 'fun'], 'times': [p0, p1]})
    [['have', 'fun'], ['Just']]
    >>> p0  # input lists should not be mutated
    [5, 1, 3]
    >>> p1
    [4, 1, 6]
    """
    check_words_and_times(words_and_times)  # verify that the input is properly formed
    words, times = words_and_times['words'], words_and_times['times']
    player_indices = range(len(times))  # contains an *index* for each player
    word_indices = range(len(words))    # contains an *index* for each word
    # BEGIN PROBLEM 10
    "*** YOUR CODE HERE ***"
    # END PROBLEM 10


def check_words_and_times(words_and_times):
    """Check that words_and_times is a {'words': words, 'times': times} dictionary
    in which each element of times is a list of numbers the same length as words.
    """
    assert 'words' in words_and_times and 'times' in words_and_times and len(words_and_times) == 2
    words, times = words_and_times['words'], words_and_times['times']
    assert all([type(w) == str for w in words]), "words should be a list of strings"
    assert all([type(t) == list for t in times]), "times should be a list of lists"
    assert all([isinstance(i, (int, float)) for t in times for i in t]), "times lists should contain numbers"
    assert all([len(t) == len(words) for t in times]), "There should be one word per time."


def get_time(times, player_num, word_index):
    """Return the time it took player_num to type the word at word_index,
    given a list of lists of times returned by time_per_word."""
    num_players = len(times)
    num_words = len(times[0])
    assert word_index < len(times[0]), f"word_index {word_index} outside of 0 to {num_words-1}"
    assert player_num < len(times), f"player_num {player_num} outside of 0 to {num_players-1}"
    return times[player_num][word_index]


enable_multiplayer = False  # Change to True when you're ready to race.

##########################
# Command Line Interface #
##########################


def run_typing_test(topics):
    """Measure typing speed and accuracy on the command line."""
    paragraphs = lines_from_file("data/sample_paragraphs.txt")
    random.shuffle(paragraphs)
    select = lambda p: True
    if topics:
        select = about(topics)
    i = 0
    while True:
        source = pick(paragraphs, select, i)
        if not source:
            print("No more paragraphs about", topics, "are available.")
            return
        print("Type the following paragraph and then press enter/return.")
        print("If you only type part of it, you will be scored only on that part.\n")
        print(source)
        print()

        start = datetime.now()
        typed = input()
        if not typed:
            print("Goodbye.")
            return
        print()

        elapsed = (datetime.now() - start).total_seconds()
        print("Nice work!")
        print("Words per minute:", wpm(typed, elapsed))
        print("Accuracy:        ", accuracy(typed, source))

        print("\nPress enter/return for the next paragraph or type q to quit.")
        if input().strip() == "q":
            return
        i += 1


@main
def run(*args):
    """Read in the command-line argument and calls corresponding functions."""
    import argparse

    parser = argparse.ArgumentParser(description="Typing Test")
    parser.add_argument("topic", help="Topic word", nargs="*")
    parser.add_argument("-t", help="Run typing test", action="store_true")

    args = parser.parse_args()
    if args.t:
        run_typing_test(args.topic)
import itertools
from copy import deepcopy


# action types
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
TYPE_45_BOMB = 45
TYPE_46_BOMB = 46
TYPE_47_BOMB = 47
TYPE_48_BOMB = 48

TYPE_5_KING_BOMB = 5
TYPE_6_STRAIGHT_FLUSH = 6

TYPE_7_3_2 = 7  # triple + pair
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10

TYPE_15_INVALID = 15

# display setting
separator_len = 96

# misc.
card2col = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 
            14: 0, 15: 1, 16: 2, 17: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 
            27: 0, 28: 1, 29: 2, 30: 3, 31: 4, 32: 5, 33: 6, 34: 7, 35: 8, 36: 9, 37: 10, 38: 11, 39: 12, 
            40: 0, 41: 1, 42: 2, 43: 3, 44: 4, 45: 5, 46: 6, 47: 7, 48: 8, 49: 9, 50: 10, 51: 11, 52: 12,
            55: 0, 56: 1, 57: 2, 58: 3, 59: 4, 60: 5, 61: 6, 62: 7, 63: 8, 64: 9, 65: 10, 66: 11, 67: 12, 
            68: 0, 69: 1, 70: 2, 71: 3, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10, 79: 11, 80: 12, 
            81: 0, 82: 1, 83: 2, 84: 3, 85: 4, 86: 5, 87: 6, 88: 7, 89: 8, 90: 9, 91: 10, 92: 11, 93: 12, 
            94: 0, 95: 1, 96: 2, 97: 3, 98: 4, 99: 5, 100: 6, 101: 7, 102: 8, 103: 9, 104: 10, 105: 11, 106: 12}

card2row = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 
            14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 
            27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 
            40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3,
            55: 4, 56: 4, 57: 4, 58: 4, 59: 4, 60: 4, 61: 4, 62: 4, 63: 4, 64: 4, 65: 4, 66: 4, 67: 4, 
            68: 5, 69: 5, 70: 5, 71: 5, 72: 5, 73: 5, 74: 5, 75: 5, 76: 5, 77: 5, 78: 5, 79: 5, 80: 5, 
            81: 6, 82: 6, 83: 6, 84: 6, 85: 6, 86: 6, 87: 6, 88: 6, 89: 6, 90: 6, 91: 6, 92: 6, 93: 6, 
            94: 7, 95: 7, 96: 7, 97: 7, 98: 7, 99: 7, 100: 7, 101: 7, 102: 7, 103: 7, 104: 7, 105: 7, 106: 7,
            53: 8, 107: 8, 54: 9, 108: 9}

row2symbol = {
    0: '\033[31m\u2665\033[0m', 1: '\033[31m\u2666\033[0m', 2: '\u2663', 3: '\u2660',
    4: '\033[31m\u2665\033[0m', 5: '\033[31m\u2666\033[0m', 6: '\u2663', 7: '\u2660',
    8: '\033[31m鬼\033[0m', 9: '鬼'
}

rank2str = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A', 20: '小', 30: '大'
}

rankDiffAndScore = {
    0: 14,
    1: 15, -1: 13,
    2: 16, -2: 12,
    3: 17, -3: 11,
    4: 18, -4: 10,
    5: 19, -5: 9,
    6: 20, -6: 8,
    7: 21, -7: 7,
    8: 22, -8: 6,
    9: 23, -9: 5,
    10: 24, -10: 4,
    11: 25, -11: 3,
    12: 26, -12: 2,
    13: 27, -13: 1,
    14: 28, -14: 0,
    15: 28, -15: 0,
}


steps2player = {  # distance between winner1 (as keys) and any of the ramining three
    1: {2: 3, 3: 2, 4: 1},
    2: {3: 3, 4: 2, 1: 1},
    3: {4: 3, 1: 2, 2: 1},
    4: {1: 3, 2: 2, 3: 1},
}

def advancex_mod4(playerNum, deltaAdv):
        result = (playerNum + deltaAdv) % 4
        if result == 0:
            result = 4
        return result


# check if move is a continuous sequence
def is_continuous_seq(move):  
    """ This is ONLY called to handle the following:
     TYPE_8_SERIAL_SINGLE, TYPE_6_ROYAL_FLUSH 
     TYPE_9_SERIAL_PAIR, TYPE_10_SERIAL_TRIPLE"""
    if move[-1] == 14 and move[0] == 2:  # Treating `A` as `1`
        if len(move) == 5 and move[1] == 3 and move[2] == 4 and move[3] == 5 \
            or len(move) == 3 and move[1] == 3 \
            or len(move) == 2:
            return True

    if len(move) == 4:  # checks straight/straight flush constructed via wild cards
        move2 = deepcopy(move)
        if move2[-1] == 14:
            if move2[0] == 2 or move2[0] == 3:
                move2 = [1] + move2[:-1]
        
        diffs = [move2[_ + 1] - move2[_] for _ in range(3)]
        if diffs.count(1) == 2 and diffs.count(2) == 1:  # unique pattern
            return True
        if diffs.count(1) == 3:
            return True
        
    i = 0
    while i < len(move) - 1:
        if move[i+1] - move[i] != 1:
            return False
        i += 1
    return True


def is_same_suit(move):
    s0mod4 = card2row[move[0]] % 4

    for _ in range(1, len(move)):
        if _ not in card2row or card2row[move[_]] % 4 != s0mod4:
            return False
    
    return True


def card2rank(card):
    m_rank = 0

    if card == 53 or card == 107:
        m_rank = 20
    elif card == 54 or card == 108:
        m_rank = 30
    else:
        m_rank = card2col[card] + 1
    
    if m_rank == 1:
        m_rank = 14
    
    return m_rank


def card2suit(card):
    m_suit = -1

    if card == 53 or card == 107:
        m_suit = 5
    elif card == 54 or card == 108:
        m_suit = 6
    else:
        m_suit = card2row[card] % 4
    
    return m_suit


def cards2rank(move):
    move_ranked = []
    for m in move:
        move_ranked.append(card2rank(m))
    
    move_ranked.sort()
    return move_ranked


# return all possible results of selecting num cards from cards list
def select(cards, num):
    return [list(i) for i in itertools.combinations(cards, num)]

# select([1,2,3], 2) --> [[1, 2], [1, 3], [2, 3]]

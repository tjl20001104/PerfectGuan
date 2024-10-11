from perfectguan.env.move_generator import MovesGenerator
from perfectguan.env import move_detector as md
from perfectguan.env.utils import card2col, card2suit, card2rank, rank2str
from perfectguan.config import ACTION_SIZE
from copy import deepcopy
from collections import Counter, defaultdict
import numpy as np
import json

def InitMatrix(N):
    matrixSize = [int(N/i+0.5)+1 for i in range(1,5)]
    F = np.zeros(matrixSize) # 1,2,3,4
    F[0,0,0,0] = 0
    for i4 in range(matrixSize[3]):
        for i3 in range(matrixSize[2]):
            for i2 in range(matrixSize[1]):
                for i1 in range(matrixSize[0]):
                    F[i1,i2,i3,i4] = i1+i2+i3+i4
                    if (4*i4+3*i3+2*i2+i1<=N):
                        if i1 != 0:
                            F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1-1,i2,i3,i4]+1)
                        if i2 != 0:
                            F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1,i2-1,i3,i4]+1)
                        if i3 != 0:
                            F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1,i2,i3-1,i4]+1)
                            if i2 != 0:
                                F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1,i2-1,i3-1,i4]+1)
                            if i3 >= 1:
                                F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1+1,i2+1,i3-1,i4])
                        if i4 != 0:
                            F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1,i2,i3,i4-1]+1)
                            if i4 >= 1:
                                F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1+1,i2,i3+1,i4-1])
                                F[i1,i2,i3,i4] = min(F[i1,i2,i3,i4],F[i1,i2+2,i3,i4-1])
                            
    return F

F = InitMatrix(30)

def min_steps_play_out(handCards,rank):
    def dfs(step, cardsNum, special_cards):
        global ans
        if step > ans:
            return
        Cnt = np.zeros(10)
        for i in range(13):
            Cnt[int(cardsNum[i])] += 1
        ans = min(ans, step+now_step(int(Cnt[1]),int(Cnt[2]),int(Cnt[3]),int(Cnt[4]),special_cards))
        for k in range(1,4):
            for i in range(-1,13):
                pos = i
                while pos <= 11 and cardsNum[pos]>= k:
                    cardsNum[pos] = cardsNum[pos] - k
                    if pos-i+1 == shunzi[k]:
                        dfs(step+1, cardsNum, special_cards)
                    pos = pos + 1
                pos = pos - 1
                while pos >= i:
                    cardsNum[pos] = cardsNum[pos] + k
                    pos = pos - 1
        return
    
    def now_step(x1,x2,x3,x4,special_cards):

        def noWildCard(x1,x2,x3,x4,black_num,red_num):
            if red_num == 0 and black_num == 0: # 0,0
                return F[x1,x2,x3,x4]
            elif red_num == 2 and black_num == 2: # 2,2
                return min(F[x1,x2+2,x3,x4],F[x1,x2,x3,x4]+1)
            elif red_num + black_num == 3: # 1,2
                return F[x1+1,x2+1,x3,x4]
            elif red_num + black_num == 1: # 0,1
                return F[x1+1,x2,x3,x4]
            elif red_num == 1 and black_num == 1: # 1,1
                return F[x1+2,x2,x3,x4]
            else: # 0,2
                return F[x1,x2+1,x3,x4]
        
        wild_num = special_cards[0]
        black_num = special_cards[1]
        red_num = special_cards[2]
        if wild_num == 0:
            return noWildCard(x1,x2,x3,x4,black_num,red_num)
        elif wild_num == 1:
            if x1 + x2 + x3 + x4 == 0:
                return noWildCard(x1,x2,x3,x4,black_num,red_num) + 1
            else:
                minstepcandidiate = []
                minstepcandidiate.append(noWildCard(x1+1,x2,x3,x4,black_num,red_num))
                if x1 > 0:
                    minstepcandidiate.append(noWildCard(x1-1,x2+1,x3,x4,black_num,red_num))
                if x2 > 0:
                    minstepcandidiate.append(noWildCard(x1,x2-1,x3+1,x4,black_num,red_num))
                if x3 > 0:
                    minstepcandidiate.append(noWildCard(x1,x2,x3-1,x4+1,black_num,red_num))
                if x4 > 0:
                    minstepcandidiate.append(noWildCard(x1,x2,x3,x4-1,black_num,red_num)+1)
                return min(minstepcandidiate)
        elif wild_num == 2:
            if x1 + x2 + x3 + x4 == 0:
                return noWildCard(x1,x2,x3,x4,black_num,red_num) + 1
            elif x1 + x2 + x3 + x4 == 1:
                minstepcandidiate = []
                minstepcandidiate.append(noWildCard(x1+2,x2,x3,x4,black_num,red_num))
                minstepcandidiate.append(noWildCard(x1,x2+1,x3,x4,black_num,red_num))
                if x1 == 1:
                    minstepcandidiate.append(noWildCard(x1-1,x2,x3+1,x4,black_num,red_num))
                if x2 == 0:
                    minstepcandidiate.append(noWildCard(x1,x2-1,x3,x4+1,black_num,red_num))
                if x3 == 0:
                    minstepcandidiate.append(noWildCard(x1,x2,x3-1,x4,black_num,red_num)+1)
                if x4 == 0:
                    minstepcandidiate.append(noWildCard(x1,x2,x3,x4-1,black_num,red_num)+1)
                return min(minstepcandidiate)
            else:
                minstepcandidiate = []
                minstepcandidiate.append(noWildCard(x1+2,x2,x3,x4,black_num,red_num))
                minstepcandidiate.append(noWildCard(x1,x2+1,x3,x4,black_num,red_num))
                if x1 >= 1:
                    minstepcandidiate.append(noWildCard(x1-1,x2,x3+1,x4,black_num,red_num))
                    if x1 >= 2:
                        minstepcandidiate.append(noWildCard(x1-2,x2+2,x3,x4,black_num,red_num))
                    if x2 > 0:
                        minstepcandidiate.append(noWildCard(x1-1,x2,x3+1,x4,black_num,red_num))
                    if x3 > 0:
                        minstepcandidiate.append(noWildCard(x1-1,x2+1,x3-1,x4+1,black_num,red_num))
                    if x4 > 0:
                        minstepcandidiate.append(noWildCard(x1-1,x2+1,x3,x4-1,black_num,red_num)+1)

                if x2 >= 1:
                    minstepcandidiate.append(noWildCard(x1,x2-1,x3,x4+1,black_num,red_num))
                    if x2 >= 2:
                        minstepcandidiate.append(noWildCard(x1,x2-2,x3+2,x4,black_num,red_num))
                    if x3 > 0:
                        minstepcandidiate.append(noWildCard(x1,x2-1,x3,x4+1,black_num,red_num))
                    if x4 > 0:
                        minstepcandidiate.append(noWildCard(x1,x2-1,x3+1,x4-1,black_num,red_num)+1)
                
                if x3 >= 1:
                    minstepcandidiate.append(noWildCard(x1,x2,x3-1,x4,black_num,red_num)+1)
                    if x3 >= 2:
                        minstepcandidiate.append(noWildCard(x1,x2,x3-2,x4+2,black_num,red_num))
                    if x4 > 0:
                        minstepcandidiate.append(noWildCard(x1,x2,x3-1,x4,black_num,red_num)+1)
                
                if x4 >= 1:
                    minstepcandidiate.append(noWildCard(x1,x2,x3,x4-1,black_num,red_num)+1)
                    if x4 >= 2:
                        minstepcandidiate.append(noWildCard(x1,x2,x3,x4-2,black_num,red_num)+2)

                return min(minstepcandidiate)
    
    # [2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A] + [rank, B, R]
    cardsNum = [0] * 13
    special_cards = [0] * 3
    shunzi = [0, 5, 3, 2]
    inf = 1000

    for card in handCards:
        if card == 53 or card == 107:
            special_cards[1] += 1
        elif card == 54 or card == 108:
            special_cards[2] += 1
        else:
            c_rank = card2col[card]
            c_suit = card2suit(card)
            if c_rank == 0:
                c_rank = 13
            if c_rank + 1 == rank and c_suit == 0:
                special_cards[0] += 1
            else:
                cardsNum[c_rank - 1] += 1
    
    global ans
    ans = inf
    
    if special_cards[0] >= 0:
        dfs(0, cardsNum, special_cards)
    if special_cards[0] >= 1:
        tmpSpecialCards = deepcopy(special_cards)
        tmpSpecialCards[0] -= 1
        for i in range(13):
            tmpCardsNum = deepcopy(cardsNum)
            tmpCardsNum[i] += 1
            dfs(0,tmpCardsNum,tmpSpecialCards)
    elif special_cards[0] >= 2:
        tmpSpecialCards = deepcopy(special_cards)
        tmpSpecialCards[0] -= 2
        for i in range(13):
            for j in range(13):
                tmpCardsNum = deepcopy(cardsNum)
                tmpCardsNum[i] += 1
                tmpCardsNum[j] += 1
                dfs(0,tmpCardsNum,tmpSpecialCards)
    return ans

def get_wild_cards(handCards, rank):
    result = []
    for card in handCards:
        if card2rank(card) == rank and card2suit(card) % 4 == 0:
            result.append(card)
    return result

def get_legal_actions(handCards,rank):
    mg = MovesGenerator(handCards)
    wild_cards_ = get_wild_cards(handCards, rank)
    moves = mg.gen_moves()
    if len(wild_cards_) > 0:
        for reps_ in range(2, 9):
            moves += mg.gen_cards_reps_filtered_w(wild_cards_, -1, reps_)

        moves += mg.gen_cards_straight_filtered_w(wild_cards_, -1)
        moves += mg.gen_cards_straight_flush_filtered_w(wild_cards_, -1)
        moves += mg.gen_cards_steel_plate_filtered_w(wild_cards_, -1)
        moves += mg.gen_cards_wooden_plate_filtered_w(wild_cards_, -1)
        moves += mg.gen_type_7_3_2_filtered_w(wild_cards_, -1)
    return moves

suit2str = {0:'H', 1:'D', 2:'C', 3:'S', 5:'S', 6:'H'}
rank2str = {
    1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: 'H', 20: 'B', 30: 'R'}
rank2col = {1:12, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8, 11:9, 
            12:10, 13:11, 14:12, 20:0, 30:1}
with open('action_space.json') as f:
    action_space = json.load(f)

def action2specific(action, rank):
    rank_array = np.zeros(13)
    rank_array[rank - 2] = 1
    if action == []:
        action_array = np.zeros((ACTION_SIZE-13))
        action_array = np.hstack([action_array,rank_array])
        return action_array, action_space['PASS']
    action_type = md.get_move_type(action, rank2play=rank)
    a_type = action_type['type']
    a_rank = action_type.get('rank', -1)
    action_suit = ''
    wild_usage = (rank in action) + ((rank+54) in action)

    # SINGLE
    if a_type == 1:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank]
    # DOUBLE
    elif a_type == 2:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 2
    # TRIPLE
    elif a_type == 3:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 3
    # FOUR BOMB
    elif a_type == 4:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 4
    # FIVE BOMB
    elif a_type == 45:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 5
    # SIX BOMB
    elif a_type == 46:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 6
    # SEVEN BOMB
    elif a_type == 47:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 7
    # EIGHT BOMB
    elif a_type == 48:
        if a_rank == 17:
            a_rank = rank
        a_list = [a_rank] * 8
    # ROCKET
    elif a_type == 5:
        a_list = [20,20,30,30]
    # STRAIGHT FLUSH
    elif a_type == 6:
        suit_list = [card2suit(a) for a in action]
        action_suit = Counter(suit_list).most_common(1)[0][0]
        action_suit = suit2str[action_suit]
        a_list = [a_rank+i for i in range(5)]
    # TRIPLE WITH PAIR
    elif a_type == 7:
        action_ = action
        action_ = [card2rank(a) for a in action_]
        for wild_id in [rank,rank+54]:
            if wild_id in action_:
                action_.remove(wild_id)
        if a_rank == 20 or a_rank == 30:
            while a_rank in action_:
                action_.remove(a_rank)
            other_rank = a_rank
            a_rank = action_[0]
            a_list = [a_rank] * 3 + [other_rank] * 2
        else:
            if a_rank == 17:
                a_list = [rank] * 3
                while rank in action_:
                    action_.remove(rank)
                other_rank = action_[0]
            else:
                a_list = [a_rank] * 3
                while a_rank in action_:
                    action_.remove(a_rank)
                other_rank = action_[0]
            a_list = a_list + [other_rank] * 2
        a_list.sort()
    # SERIAL SINGLE
    elif a_type == 8:
        a_list = [a_rank+i for i in range(5)]
    # SERIAL DOUBLE
    elif a_type == 9:
        a_list = [a_rank+i for i in range(3)] * 2
        a_list.sort()
    # SERIAL TRIPLE
    elif a_type == 10:
        a_list = [a_rank+i for i in range(2)] * 3
        a_list.sort()

    a_matrix = np.zeros((13,8)) # 2->A
    king_matrix = np.zeros((2,2)) # B->R
    a_count = Counter(a_list)
    for k,c in a_count.items():
        if k <= 14:
            a_matrix[rank2col[k],:c] = np.ones((c))
        elif k == 20 or k == 30:
            king_matrix[rank2col[k],:c] = np.ones((c))
    wild_array = np.zeros((2))
    if wild_usage > 0:
        wild_array[:wild_usage] = 1
    if a_type in [4,45,46,47,48,5,6]:
        if a_type == 6:
            is_straight_flush = np.array([1])
        else:
            is_straight_flush = np.array([0])
        is_bomb = np.array([1])
    else:
        is_bomb = np.array([0])
        is_straight_flush = np.array([0])
    action_array = np.hstack([
        a_matrix.flatten(),
        king_matrix.flatten(),
        wild_array,
        is_bomb,
        is_straight_flush,
        rank_array
    ])
    
    a_str = ''.join([rank2str[a] for a in a_list])
    if action_suit != '':
        a_str = a_str + '.' + action_suit
    a_id = action_space[a_str]
    return action_array, a_id

def actions_merge(legal_actions, rank):
    merged_res = defaultdict(list)
    for action in legal_actions:
        action_array, a_id = action2specific(action,rank)
        flag = 1
        for a in merged_res[a_id]:
            if np.array_equal(a, action_array):
                flag = 0
        if flag == 1:
            merged_res[a_id].append(action_array)
    res = []
    for a_id,a_arrays in merged_res.items():
        for a_array in a_arrays:
            res.append((a_id,a_array))
    return res

def decode_action(action_id, legal_actions, rank):
    candidate_action = defaultdict(list)
    for action in legal_actions:
        _, a_id = action2specific(action,rank)
        if a_id == action_id:
            wild_usage = (rank in action) + ((rank+54) in action)
            candidate_action[wild_usage].append(action)
    final_action = None
    if 0 in candidate_action.keys():
        random_size = len(candidate_action[0])
        final_action = candidate_action[0][np.random.randint(random_size)]
    else:
        if 1 in candidate_action.keys():
            min_rank_sum = 1000
            for a1 in candidate_action[1]:
                a1_without_wild = deepcopy(a1)
                if rank in a1_without_wild:
                    a1_without_wild.remove(rank)
                if (rank+54) in a1_without_wild:
                    a1_without_wild.remove(rank+54)
                a1_rank = [card2rank(c) for c in a1_without_wild]
                rank_sum = np.sum(a1_rank)
                if rank_sum < min_rank_sum:
                    min_rank_sum = rank_sum
                    final_action = a1
        else:
            min_rank_sum = 1000
            for a2 in candidate_action[2]:
                a2_without_wild = deepcopy(a2)
                if rank in a2_without_wild:
                    a2_without_wild.remove(rank)
                if (rank+54) in a2_without_wild:
                    a2_without_wild.remove(rank+54)
                a2_rank = [card2rank(c) for c in a2_without_wild]
                rank_sum = np.sum(a2_rank)
                if rank_sum < min_rank_sum:
                    min_rank_sum = rank_sum
                    final_action = a2
    
    return final_action
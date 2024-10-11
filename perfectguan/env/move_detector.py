from perfectguan.env.utils import *
import collections
from perfectguan.env.utils import cards2rank, is_continuous_seq, is_same_suit
# from copy import deepcopy


# return the type of the move
def get_move_type(move_, rank2play=-1):
    if len(move_) == 0 \
        or len(move_) == 1 and isinstance(move_[0], list) and len(move_[0]) == 0:
        return {'type': TYPE_0_PASS}

    move = cards2rank(move_)
    move_size = len(move)
    move_dict = collections.Counter(move)
    count_dict = collections.defaultdict(int)
    for _, n in move_dict.items():
        count_dict[n] += 1

    move_no_wild_card = []
    if rank2play != -1:
        for m_ind in range(move_size):
            m_card = move_[m_ind]
            if card2rank(m_card) == rank2play and card2suit(m_card) % 4 == 0:
                continue
            move_no_wild_card.append(m_card)
    
    if 0 < len(move_no_wild_card) < move_size:  # at least 1 wild card is present
        if move_size == 2:
            move_nwc_rank = card2rank(move_no_wild_card[0])
            return {'type': TYPE_2_PAIR, 'rank': 17 if move_nwc_rank == rank2play else move_nwc_rank}
        elif move_size > 2:
            move_no_wild_card_ranked = cards2rank(move_no_wild_card)
            move_no_wild_card_dict = collections.Counter(move_no_wild_card_ranked)
            type_name = TYPE_15_INVALID
            moves_rank = -1

            # print('move_size: {}, move_copy_ranked: {}'.format(move_size, move_copy_ranked))
            if len(move_no_wild_card_dict) == 1:
                if move_size == 3:
                    type_name = TYPE_3_TRIPLE
                elif move_size == 4:
                    type_name = TYPE_4_BOMB
                elif move_size == 5:
                    type_name = TYPE_45_BOMB
                elif move_size == 6:
                    type_name = TYPE_46_BOMB
                elif move_size == 7:
                    type_name = TYPE_47_BOMB
                elif move_size == 8:
                    type_name = TYPE_48_BOMB

                moves_rank = 17 if move_no_wild_card_ranked[0] == rank2play else move_no_wild_card_ranked[0]
            
            elif len(move_no_wild_card_dict) == 2:  # steel plate or 7_3_2
                if move_size == 6:  # steel plate
                    if abs(move_no_wild_card_ranked[0] - move_no_wild_card_ranked[-1]) == 1:
                        type_name = TYPE_10_SERIAL_TRIPLE
                        moves_rank = move_no_wild_card_ranked[0]
                    elif move_no_wild_card_ranked[0] == 2 and move_no_wild_card_ranked[-1] == 14:
                        type_name = TYPE_10_SERIAL_TRIPLE
                        moves_rank = 1
                elif move_size == 5:  # 7_3_2
                    type_name = TYPE_7_3_2
                    if len(move_no_wild_card) == 4:
                        for _ in move_no_wild_card_dict:
                            if move_no_wild_card_dict[_] == 3:
                                moves_rank = 17 if _ == rank2play else _
                                break
                        
                        if min(move_no_wild_card_dict.values()) == 2 \
                            and max(move_no_wild_card_dict.values()) == 2:
                            moves_rank = 17 if rank2play in move_no_wild_card_dict else max(move_no_wild_card_dict.keys())

                    elif len(move_no_wild_card) == 3:
                        moves_rank = 17 if rank2play in move_no_wild_card_dict else  max(move_no_wild_card_dict.keys())

            elif len(move_no_wild_card_dict) == 3:  # wooden plate
                if is_continuous_seq(sorted(move_no_wild_card_dict.keys())):
                    if move_size == 6:
                        if len(move_no_wild_card_ranked) == 5 or len(move_no_wild_card_ranked) == 4:
                            type_name = TYPE_9_SERIAL_PAIR
                            if move_no_wild_card_ranked[0] == 2 and move_no_wild_card_ranked[-1] == 14:
                                moves_rank = 1
                            else:
                                moves_rank = move_no_wild_card_ranked[0]

            elif len(move_no_wild_card_dict) == 4:  # straight/straight flush
                is_continuous_seq([2,3,4,14])
                if is_continuous_seq(move_no_wild_card_ranked):
                    type_name = TYPE_6_STRAIGHT_FLUSH if is_same_suit(move_no_wild_card) else TYPE_8_SERIAL_SINGLE
                    if 14 in move_no_wild_card_ranked \
                        and (move_no_wild_card_ranked[0] == 2 or move_no_wild_card_ranked[0] == 3):
                        moves_rank = 1
                    else:
                        moves_rank = 10 if move_no_wild_card_ranked[0] == 11 else move_no_wild_card_ranked[0]
            
            if type_name != TYPE_15_INVALID:
                return {'type': type_name, 'rank': moves_rank}
            
    ### type = repeats?
    moves_rank = 17 if move[0] == rank2play else move[0]

    if move_size == 1:
        return {'type': TYPE_1_SINGLE, 'rank': move[0]}

    elif move_size == 2:
        if move[0] == move[1]:
            return {'type': TYPE_2_PAIR, 'rank': moves_rank}

    elif move_size == 3:
        if len(move_dict) == 1:
            return {'type': TYPE_3_TRIPLE, 'rank': moves_rank}

    elif move_size == 4:
        if len(move_dict) == 1:
            return {'type': TYPE_4_BOMB,  'rank': moves_rank}
        
        if move == [20, 20, 30, 30]:
            return {'type': TYPE_5_KING_BOMB}
        
    elif move_size == 5:
        if len(move_dict) == 1:
            return {'type': TYPE_45_BOMB, 'rank': moves_rank}
        elif len(move_dict) == 2:
            return {'type': TYPE_7_3_2, 'rank': 17 if move[2] == rank2play else move[2]}
        elif len(move_dict) == 5:
            if is_continuous_seq(move):  # only size of 5 is allowed
                if is_same_suit(move_):
                    if 14 in move and (move[0] == 2 or move[0] == 3):
                        return {'type': TYPE_6_STRAIGHT_FLUSH, 'rank': 1}
                    return {'type': TYPE_6_STRAIGHT_FLUSH, 'rank': move[0]}
                else:
                    if 14 in move and (move[0] == 2 or move[0] == 3):
                        return {'type': TYPE_8_SERIAL_SINGLE, 'rank': 1}
                    return {'type': TYPE_8_SERIAL_SINGLE, 'rank': move[0]}

    elif move_size == 6:
        if len(move_dict) == 1:
            return {'type': TYPE_46_BOMB, 'rank': moves_rank}
        
        mdkeys = sorted(move_dict.keys())
        if len(move_dict) == count_dict.get(2) and is_continuous_seq(mdkeys):
            if 14 in mdkeys and (2 in mdkeys or 3 in mdkeys):
                return {'type': TYPE_9_SERIAL_PAIR, 'rank': 1}
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': mdkeys[0]}

        if len(move_dict) == count_dict.get(3) and is_continuous_seq(mdkeys):
            if 14 in mdkeys and 2 in mdkeys:
                return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': 1}
            return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': mdkeys[0]}
    
    elif move_size == 7:
        if len(move_dict) == 1:
            return {'type': TYPE_47_BOMB, 'rank': moves_rank}
        
    elif move_size == 8:
        if len(move_dict) == 1:
            return {'type': TYPE_48_BOMB, 'rank': moves_rank}

    return {'type': TYPE_15_INVALID}

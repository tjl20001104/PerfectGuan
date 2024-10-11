from perfectguan.env.utils import select, card2rank, cards2rank, card2suit
import collections
import itertools
import functools
import numpy as np


class MovesGenerator(object):
    """
    This is for generating the possible combinations
    """
    def __init__(self, cards_list):
        self.cards_list = cards_list
        
        self.cards_dict = collections.defaultdict(int)

        for i in cards2rank(self.cards_list):
            self.cards_dict[i] += 1

        self.rankAndCards = {}
        self.rankAndSuits = {}

        for c in self.cards_list:
            c_rank = card2rank(c)
            if c_rank in self.rankAndCards:
                self.rankAndCards[c_rank].append(c)
            else:
                self.rankAndCards[c_rank] = [c]

            c_suit = card2suit(c)
            if c_rank in self.rankAndSuits:
                self.rankAndSuits[c_rank].append(c_suit)
            else:
                self.rankAndSuits[c_rank] = [c_suit]

        self.single_cards = sorted(self.rankAndCards.keys())

        self.pair_moves = []
        self.triple_moves = []

        self.gen_cards_reps()
        self.gen_type_5_king_bomb()

    def _gen_serial_moves(self, seq_len, reps):
        seq_records = list()
        moves = list()

        start = i = 0
        longest = 1
        while i < len(self.single_cards):
            if i + 1 < len(self.single_cards) and self.single_cards[i + 1] - self.single_cards[i] == 1 \
                and len(self.rankAndCards[self.single_cards[i]]) >= reps \
                and len(self.rankAndCards[self.single_cards[i + 1]]) >= reps:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        for seq in seq_records:
            if seq[1] >= seq_len:
                start, end = seq[0], seq[0] + seq[1] - 1

                while start + seq_len - 1 <= end:
                    cands = []
                    for r in range(seq_len):
                        cands_pre = self.rankAndCards[self.single_cards[start+r]]
                        if reps > 1:
                            cands_pre = select(cands_pre, reps)
                        cands.append(cands_pre)
                    
                    if reps > 1:
                        moves_pre = [p for p in itertools.product(*cands)]
                        
                        moves_pre_ = []
                        for mp in moves_pre:
                            moves_pre_.append(functools.reduce(lambda x, y: x + y, mp))
                        
                        moves.extend(moves_pre_)
                    else:
                        moves.extend([list(p) for p in itertools.product(*cands)])
                    start += 1
        
        ### treat aces as 1 ###
        if 14 in self.single_cards and len(self.rankAndCards[14]) >= reps:
            if self.single_cards[seq_records[0][0]] == 2 and seq_records[0][1] >= seq_len - 1:
                cands = []
                if reps == 1:
                    cands.append(self.rankAndCards[14])
                    for r in range(4):
                        cands.append(self.rankAndCards[self.single_cards[r]])
                    
                    moves.extend([list(p) for p in itertools.product(*cands)])
                    if [2, 30, 81, 82, 83] in moves:
                        breakpoint()
                elif reps == 2 or reps == 3:
                    cands.append(select(self.rankAndCards[14], reps))
                    for r in range(seq_len - 1):
                        cands.append(select(self.rankAndCards[self.single_cards[r]], reps))
                    
                    moves_pre = [p for p in itertools.product(*cands)]
                    
                    moves_pre_ = []
                    for mp in moves_pre:
                        moves_pre_.append(functools.reduce(lambda x, y: x + y, mp))
                    
                    moves.extend(moves_pre_)

        return moves

    def gen_cards_reps(self, reps=2):  # reps = 2: pair; 3: triple/trio; 4: bomb; etc.
        if reps == 1:
            return [[s] for s in self.cards_list]

        result = []
        for _ in self.rankAndCards:
            cands0 = self.rankAndCards[_]
            if len(cands0) >= reps:
                result.extend(cands for cands in select(cands0, reps))

        if reps == 2:
            self.pair_moves = result
        elif reps == 3:
            self.triple_moves = result

        return result

    def gen_cards_reps_filtered_w(self, wild_cards, rival_card_rank, reps):  # reps = 2: pair; 3: triple/trio; 4: bomb; etc.
        # Here we assume wild_cards is NOT empty
        result = []
        
        # num_wild_cards = len(wild_cards)

        for _ in self.rankAndCards:
            if rival_card_rank < _ < 17:  # can't pair with jokers or rank cards, AND must beat rival
                cands0 = [c0 for c0 in self.rankAndCards[_] if c0 not in wild_cards]
                if len(cands0) >= reps - 1:
                    for wc in wild_cards:
                        result.extend(cands + [wc] for cands in select(cands0, reps - 1))

                # if reps >= 3 and num_wild_cards == 2 and len(cands0) >= reps - 2:
                #     result.extend(cands + wild_cards for cands in select(cands0, reps - 2))

        return result

    def gen_cards_straight_filtered_w(self, wild_cards, rival_card_rank):
        # Here we assume wild_cards is NOT empty
        result = []
        
        # num_wild_cards = len(wild_cards)
        for wc in wild_cards:
            for _ in range(max(rival_card_rank + 1, 1), 11):
                tmp_result = []
                gaps = 0

                for i_rank in [14, 2, 3, 4, 5] if _ == 1 else [_, _ + 1, _ + 2, _ + 3, _ + 4]:
                    cands0 = [c0 for c0 in self.rankAndCards.get(i_rank, []) if c0 not in wild_cards]
                    if gaps >= 2:
                        tmp_result = []
                        break
                    
                    if gaps == 0 and len(cands0) == 0:
                        tmp_result.append([wc])
                        gaps += 1
                        continue
                    
                    tmp_result.append(cands0)

                if gaps == 1 and len(tmp_result) == 5:
                    # print(tmp_result)
                    result.extend([list(p) for p in itertools.product(*tmp_result)])

        return result
    
    def gen_cards_straight_flush_filtered_w(self, wild_cards, rival_card_rank):
        # Here we assume wild_cards is NOT empty
        result = []
        
        # num_wild_cards = len(wild_cards)
        for wc in wild_cards:
            for suit in [0,1,2,3]:
                for _ in range(max(rival_card_rank + 1, 1), 11):
                    tmp_result = []
                    gaps = 0

                    for i_rank in [14, 2, 3, 4, 5] if _ == 1 else [_, _ + 1, _ + 2, _ + 3, _ + 4]:
                        cands0 = [c0 for c0 in self.rankAndCards.get(i_rank, []) if c0 not in wild_cards and card2suit(c0) == suit]
                        if gaps >= 2:
                            tmp_result = []
                            break
                        
                        if gaps == 0 and len(cands0) == 0:
                            tmp_result.append([wc])
                            gaps += 1
                            continue
                        
                        tmp_result.append(cands0)

                    if gaps == 1 and len(tmp_result) == 5:
                        result.extend([list(p) for p in itertools.product(*tmp_result)])

        return result
    

    def gen_cards_steel_plate_filtered_w(self, wild_cards, rival_card_rank):
        result = []
        
        triple_no_wild = []
        for _ in self.rankAndCards:
            if _ > rival_card_rank:
                cands0 = [c0 for c0 in self.rankAndCards[_] if c0 not in wild_cards]
                if len(cands0) >= 3:
                    triple_no_wild.extend(cands for cands in select(cands0, 3))

        if len(triple_no_wild) > 0:
            for wc in wild_cards:
                for t in triple_no_wild:
                    t_rank = card2rank(t[0])
                    search_range = [t_rank - 1, t_rank + 1]
                    if t_rank == 2:
                        search_range[0] = 14
                    elif t_rank == 14:
                        search_range[1] = 2
                    
                    for sr in search_range:
                        if sr > rival_card_rank and sr in self.rankAndCards:
                            if rival_card_rank >= 1:
                                if t_rank == 2 and sr == 14 or t_rank == 14 and sr == 2:
                                    continue

                            cands1 = [c1 for c1 in self.rankAndCards[sr] if c1 not in wild_cards]
                            if len(cands1) == 2:
                                result.append(cands1 + [wc] + t)

        # print('result of steel plate w/ wild card: {}\n'.format(result))
        return result


    def gen_cards_wooden_plate_filtered_w(self, wild_cards, rival_card_rank):
        result = []
        
        # use **1** wild card at a time
        for wc in wild_cards:
            for _ in self.rankAndCards:
                if rival_card_rank >= 1 and _ == 14:
                    continue
                
                if _ > rival_card_rank:
                    r2 = 2 if _ == 14 else _ + 1
                    r3 = 3 if _ == 14 else _ + 2
                    
                    if r2 not in self.single_cards or r3 not in self.single_cards:
                        continue

                    cands1 = [c1 for c1 in self.rankAndCards[_] if c1 not in wild_cards]
                    cands2 = [c2 for c2 in self.rankAndCards[r2] if c2 not in wild_cards]
                    cands3 = [c3 for c3 in self.rankAndCards[r3] if c3 not in wild_cards]

                    l1 = len(cands1)
                    l2 = len(cands2)
                    l3 = len(cands3)
                    l123 = [l1, l2, l3]

                    if l123.count(0) > 0 or l123.count(1) != 1:
                        continue
                    
                    cands1 = [cands1 + [wc]] if l1 == 1 else select(cands1, 2)
                    cands2 = [cands2 + [wc]] if l2 == 1 else select(cands2, 2)
                    cands3 = [cands3 + [wc]] if l3 == 1 else select(cands3, 2)
                    cands123 = [cands1, cands2, cands3]

                    result.extend([p[0]+p[1]+p[2] for p in itertools.product(*cands123)])

        # Use BOTH wild cards at once
        if len(wild_cards) == 2:
            for _ in self.rankAndCards:
                if _ > rival_card_rank:
                    r2 = 2 if _ == 14 else _ + 1
                    r3 = 3 if _ == 14 else _ + 2
                    
                    if r2 not in self.single_cards or r3 not in self.single_cards:
                        continue

                    cands1 = [c1 for c1 in self.rankAndCards[_] if c1 not in wild_cards]
                    cands2 = [c2 for c2 in self.rankAndCards[r2] if c2 not in wild_cards]
                    cands3 = [c3 for c3 in self.rankAndCards[r3] if c3 not in wild_cards]

                    l1 = len(cands1)
                    l2 = len(cands2)
                    l3 = len(cands3)
                    l123 = [l1, l2, l3]

                    if l123.count(0) > 0 or l123.count(1) != 2:
                        continue
                    
                    cands1 = [cands1 + [wild_cards[0]]] if l1 == 1 else select(cands1, 2)
                    cands2 = [cands2 + ([wild_cards[1]] if l1 == 1 else [wild_cards[0]])] if l2 == 1 else select(cands2, 2)
                    cands3 = [cands3 + [wild_cards[1]]] if l3 == 1 else select(cands3, 2)
                    cands123 = [cands1, cands2, cands3]

                    result.extend([p[0]+p[1]+p[2] for p in itertools.product(*cands123)])

        # print('result of wooden plate w/ wild card: {}\n'.format(result))
        return result
    

    def gen_type_5_king_bomb(self):
        result = []
        if 53 in self.cards_list and 54 in self.cards_list \
            and 107 in self.cards_list and 108 in self.cards_list:
            result = [[53, 54, 107, 108]]
        return result
    
    def gen_straight_flush(self):
        seq_records = list()
        moves = list()

        for suit in [0,1,2,3]:
            start = i = 0
            longest = 1
            while i < len(self.single_cards):
                if i + 1 < len(self.single_cards) and self.single_cards[i + 1] - self.single_cards[i] == 1 \
                    and suit in self.rankAndSuits[self.single_cards[i]] \
                        and suit in self.rankAndSuits[self.single_cards[i + 1]]:  # i and i + 1 have the same suit
                    longest += 1
                    i += 1
                else:
                    seq_records.append((start, longest))
                    i += 1
                    start = i
                    longest = 1

            if 14 in self.single_cards and suit in self.rankAndSuits[14]\
                and self.single_cards[seq_records[0][0]] == 2 and seq_records[0][1] >= 4:
                cands = []
                cands.append([cp for cp in self.rankAndCards[14] if card2suit(cp) == suit])
                
                for r in range(4):
                    cands_pre = self.rankAndCards[self.single_cards[r]]
                    cands.append([cp for cp in cands_pre if card2suit(cp) == suit])

                moves.extend([list(p) for p in itertools.product(*cands)])

            for seq in seq_records:
                if seq[1] >= 5:
                    start, end = seq[0], seq[0] + seq[1] - 1

                    while start + 4 <= end:
                        cands = []
                        for r in range(5):
                            cands_pre = self.rankAndCards[self.single_cards[start+r]]
                            cands.append([cp for cp in cands_pre if card2suit(cp) == suit])

                        moves.extend([list(p) for p in itertools.product(*cands)])
                        start += 1
        
        return moves

    def gen_type_7_3_2(self):
        result = list()
        # print('pair_moves: {}\ntriple_moves: {}\n'.format(self.pair_moves, self.triple_moves))
        # print('732 cands?', end='')

        if len(self.pair_moves) == 0:
            self.gen_cards_reps(2)

        if len(self.triple_moves) == 0:
            self.gen_cards_reps(3)

        for p in self.pair_moves:
            for t in self.triple_moves:
                if card2rank(p[0]) != card2rank(t[0]):
                    # print(p+t)
                    result.append(p+t)
        # print('732 finishing gen..')
        return result

    def gen_type_7_3_2_filtered_w(self, wild_cards, rival_card_rank):
        result = list()
        # print('pair_moves: {}\ntriple_moves: {}\n'.format(self.pair_moves, self.triple_moves))
        # print('732 cands?', end='')

        pair_moves_no_wild_card = []
        triple_moves_no_wild_card = []
        for _ in self.rankAndCards:
            cands0 = self.rankAndCards[_]
            cands0 = [c0 for c0 in cands0 if c0 not in wild_cards]
            if len(cands0) >= 2:
                pair_moves_no_wild_card.extend(cands for cands in select(cands0, 2))
            
            if len(cands0) >= 3 and _ > rival_card_rank:
                triple_moves_no_wild_card.extend(cands for cands in select(cands0, 3))

        if len(pair_moves_no_wild_card) > 0:
            triple_moves_w = self.gen_cards_reps_filtered_w(wild_cards, rival_card_rank, 3)
            if len(triple_moves_w) > 0:
                for p in pair_moves_no_wild_card:
                    for t in triple_moves_w:
                        t0_rank = card2rank(t[0])
                        if card2rank(p[0]) != t0_rank and t0_rank > rival_card_rank:
                            result.append(p+t)

        if len(triple_moves_no_wild_card) > 0:
            pair_moves_w = self.gen_cards_reps_filtered_w(wild_cards, -1, 2)
            if len(pair_moves_w) > 0:
                for p in pair_moves_w:
                    for t in triple_moves_no_wild_card:
                        if card2rank(p[0]) != card2rank(t[0]):
                            result.append(p+t)

        return result
    
    # generate all possible moves from given cards
    def gen_moves(self):
        moves = []
        
        for reps in range(1, 9):
            moves.extend(self.gen_cards_reps(reps=reps))

        moves.extend(self.gen_type_5_king_bomb())
        moves.extend(self.gen_type_7_3_2())
        
        moves.extend(self._gen_serial_moves(5, 1))
        moves.extend(self._gen_serial_moves(3, 2))
        moves.extend(self._gen_serial_moves(2, 3))

        return moves

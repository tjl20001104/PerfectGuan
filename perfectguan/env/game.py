from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGenerator
from .utils import card2rank, card2suit, card2row, row2symbol, rank2str, \
      rankDiffAndScore, advancex_mod4, separator_len, steps2player
import numpy as np
import pickle
from os import path
from perfectguan.config import *
import collections
import random

class GameEnv(object):
    def __init__(self, players):
        self.card_play_action_seq = []

        # self.three_landlord_cards = None  # tribute # placeholder
        self.game_over = False

        self.acting_player_position = None

        # self.player_utility_dict = None  # TypeError: 'NoneType' object does not support item assignment
        self.player_utility_dict = {}  # Need to specify: dict type

        self.players = players

        self.last_move_dict = {'p1': [],
                               'p2': [],
                               'p3': [],
                               'p4': []}

        self.played_cards = {'p1': [],
                             'p2': [],
                             'p3': [],
                             'p4': []}

        self.last_move = []
        self.last_three_moves = []

        self.info_sets = {'p1': InfoSet('p1'),
                         'p2': InfoSet('p2'),
                         'p3': InfoSet('p3'),
                         'p4': InfoSet('p4')}

        self.last_pid = 'p1'  # pid player ID
        self.is_new_round = 0

        self.first_player = 1  # who goes first
        self.finishingOrder = []

        self.playerAndRank = { 'p1': 2, 'p2': 2, 'p3': 2, 'p4': 2 }
        self.playerAndScore = { 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0 }
        self.prev_finishingOrder = []
        
        if PICKLE:
            if path.isfile('playerAndRank'):
                f=open("playerAndRank","rb")
                self.playerAndRank=pickle.load(f)
                f.close()
            
            if path.isfile('playerAndScore'):
                f=open("playerAndScore","rb")
                self.playerAndScore=pickle.load(f)
                f.close()
            
            if path.isfile('prev_finishingOrder'):
                f=open("prev_finishingOrder","rb")
                try:
                    self.prev_finishingOrder=pickle.load(f)
                except EOFError:
                    print('Loading prev_finishingOrder failed, returning an empty list')
                    self.prev_finishingOrder=[]
                f.close()
        
        self.num_steps = 0
        self.w1_step = -1
        self.w2_step = -1
        self.w1_yield = True  # yield right of playing to 1st/2nd winner's partner
        self.w2_yield = True
        self.playerAndWins = { 'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0 }
        self.rank2play = 2
        self.coop_status = { 'p1': [0, 0, 0], 'p2': [0, 0, 0], 'p3': [0, 0, 0], 'p4': [0, 0, 0] }  
        # ^^ 0: not applicable; 1: applicable + crushing; 2: applicable + NOT crushing
        self.coop_counter = { 'p1': [0, 0, 0, 0, 0, 0], 'p2': [0, 0, 0, 0, 0, 0], 'p3': [0, 0, 0, 0, 0, 0], 'p4': [0, 0, 0, 0, 0, 0] }
        
        self.turns2skip = -1
        self.double_downs = 0


    def card_play_init(self, card_play_data):
        # load cards into hands
        self.info_sets['p1'].player_hand_cards = \
            card_play_data['p1']
        self.info_sets['p2'].player_hand_cards = \
            card_play_data['p2']
        self.info_sets['p3'].player_hand_cards = \
            card_play_data['p3']
        self.info_sets['p4'].player_hand_cards = \
            card_play_data['p4']
        
        # self.three_landlord_cards = card_play_data['three_landlord_cards']  # tribute # placeholder
        
        # self.get_acting_player_position()
        self.acting_player_position = playerNum2Str[self.first_player]
        self.game_infoset = self.get_infoset()


    def game_done(self):  # onGameEnd()
        num_empty_hands = 0
        players_finished = []
        if len(self.info_sets['p1'].player_hand_cards) == 0:
            num_empty_hands += 1
            players_finished.append(1)
        if len(self.info_sets['p2'].player_hand_cards) == 0:
            num_empty_hands += 1
            players_finished.append(2)
        if len(self.info_sets['p3'].player_hand_cards) == 0:
            num_empty_hands += 1
            players_finished.append(3)
        if len(self.info_sets['p4'].player_hand_cards) == 0:
            num_empty_hands += 1
            players_finished.append(4)

        if num_empty_hands == 1:
            if len(self.finishingOrder) == 0:
                self.finishingOrder.append(players_finished[0])
                if self.w1_step == -1:
                    self.w1_step = self.num_steps
        elif num_empty_hands == 2:
            if len(self.finishingOrder) == 1:
                for pf in players_finished:
                    if pf != self.finishingOrder[0]:
                        self.finishingOrder.append(pf)
                        if self.w2_step == -1:
                            self.w2_step = self.num_steps
                        break
        elif num_empty_hands == 3:  # game is over
            if len(self.finishingOrder) == 2:
                for pf in players_finished:
                    if pf != self.finishingOrder[0] and pf != self.finishingOrder[1]:
                        self.finishingOrder.append(pf)
                        break
            
            if len(self.finishingOrder) == 3:
                for last_player in [1, 2, 3, 4]:
                    if last_player not in players_finished:
                        self.finishingOrder.append(last_player)
                        break

            self.update_player_stats()
            self.game_over = True
            
            # Persist data
            if PICKLE:
                f=open("playerAndRank","wb")
                pickle.dump(self.playerAndRank, f)
                f.close()

                f=open("playerAndScore","wb")
                pickle.dump(self.playerAndScore, f)
                f.close()

                f=open("prev_finishingOrder","wb")
                pickle.dump(self.finishingOrder, f)
                f.close()
    

    def update_player_stats(self):
        winner1 = self.finishingOrder[0]
        winner2 = advancex_mod4(winner1, 2)
        loser1 = advancex_mod4(winner1, 1)
        loser2 = advancex_mod4(winner1, 3)
        
        self.playerAndWins[playerNum2Str[winner1]] += 1
        self.playerAndWins[playerNum2Str[winner2]] += 1

        lastRank_winner = self.playerAndRank[playerNum2Str[winner1]]
        lastRank_loser = self.playerAndRank[playerNum2Str[loser1]]

        nextRank = -1
        deltaRank = 0
           
        for i_fo in range(1,4):
            if self.finishingOrder[i_fo] == winner2:
                deltaRank = 4 - i_fo
                nextRank = lastRank_winner + 4 - i_fo
                break
        
        # Set the first player for the next round
        self.first_player = self.finishingOrder[-1]  # Caution: the first player to play in the next round is NOT the current winner 
        
        if nextRank > 14:
            if lastRank_winner < 14:  # Force to play rank 'A'
                self.playerAndRank[playerNum2Str[winner1]] = 14
                self.playerAndRank[playerNum2Str[winner2]] = 14
            else:
                self.first_player = 1
                self.playerAndRank = { 'p1': 2, 'p2': 2, 'p3': 2, 'p4': 2 }
        else:
            self.playerAndRank[playerNum2Str[winner1]] = nextRank
            self.playerAndRank[playerNum2Str[winner2]] = nextRank

        rankDiff = abs(lastRank_loser - nextRank)

        self.player_utility_dict[playerNum2Str[winner1]] = deltaRank
        self.player_utility_dict[playerNum2Str[winner2]] = deltaRank
        self.player_utility_dict[playerNum2Str[loser1]] = -1
        self.player_utility_dict[playerNum2Str[loser2]] = -1

        self.playerAndScore[playerNum2Str[winner1]] += rankDiffAndScore[rankDiff]
        self.playerAndScore[playerNum2Str[winner2]] += rankDiffAndScore[rankDiff]
        self.playerAndScore[playerNum2Str[loser1]] += rankDiffAndScore[-rankDiff]
        self.playerAndScore[playerNum2Str[loser2]] += rankDiffAndScore[-rankDiff]

        self.rank2play = self.playerAndRank[playerNum2Str[winner1]]
        self.prev_finishingOrder = self.finishingOrder.copy()
        if abs(self.finishingOrder[-1] - self.finishingOrder[-2]) == 2:
            self.double_downs += 1


    def str2action(self, a):
        # strip out all the white spaces in a
        a = "".join(a.split())
        la = len(a)
        action = []

        ptr = 0
        a1 = ''
        while ptr < la:
            a2 = a[ptr]
            if a2 == ',':
                try:
                    action.append(int(a1))
                except ValueError:
                    action.append(-1)
                a1 = ''
            else:
                a1 += a2
                if ptr == la - 1:
                    try:
                        action.append(int(a1))
                    except ValueError:
                        action.append(-1)

            ptr += 1
        
        action.sort()
        return action


    def get_rank2play(self):
        return self.rank2play


    def display_cards_in_hand(self, role):
        print('\nCards in your hand:')
        rankAndCards = {}
        for c in self.info_sets[role].player_hand_cards:
            c_rank = card2rank(c)
            if c_rank in rankAndCards:
                rankAndCards[c_rank].append(c)
            else:
                rankAndCards[c_rank] = [c]

        for sc in sorted(rankAndCards.keys()):
            sc_str = ('\U0001F451' if sc == self.rank2play else 'r') + rank2str[sc] + ' \u2192 ['
            for isc in rankAndCards[sc]:
                if isc == 53 or isc == 107:
                    isc_symb = '\033[31m鬼\033[0m'
                elif isc == 54 or isc == 108:
                    isc_symb = '鬼'
                else:
                    isc_symb = row2symbol[card2row[isc]]
                sc_str += str(isc) + isc_symb + ','
            sc_str = sc_str[1:len(sc_str)-1]
            
            print(sc_str + (']\U0001F4AA' if sc == self.rank2play else ']'))


    def display_tribute_info(self, donor, returner, card2donate, card2return):
        print('\n\U0001F381 {} donates {} <> {} returns {}\n'.format(
            donor, rank2str[card2rank(card2donate)] + row2symbol[card2row[card2donate]],
            returner, rank2str[card2rank(card2return)] + row2symbol[card2row[card2return]]))
        

    def get_card2return(self, cards_in_hand):
        player_rankAndCards = {}
        player_wild_cards = []

        for c in cards_in_hand:
            c_rank = card2rank(c)
            if c_rank == self.rank2play:
                if card2suit(c) % 4 == 0:
                    player_wild_cards.append(c)
            else:  # exclude wild cards
                if c_rank in player_rankAndCards:
                    player_rankAndCards[c_rank].append(c)
                else:
                    player_rankAndCards[c_rank] = [c]

        card2return = -1
        c_rank_min = min(player_rankAndCards.keys())
        if c_rank_min > 10:
            card2return = random.choice(player_rankAndCards[c_rank_min])
        else:
            player_mg = MovesGenerator(cards_in_hand)
            player_straight_flush = player_mg.gen_straight_flush()
            if len(player_wild_cards) > 0:
                player_straight_flush += player_mg.gen_cards_straight_filtered_w(player_wild_cards, -1)

            card2return_cands = []
            for cd in filter(lambda k: k <= 10, player_rankAndCards):
                for cc in player_rankAndCards[cd]:
                    if len(player_straight_flush) > 0:
                        for pf1_sf in player_straight_flush:
                            if cc not in pf1_sf:
                                card2return_cands.append(cc)
                    else:
                        card2return_cands.append(cc)
            
            card2return = random.choice(card2return_cands)
        
        return card2return


    def get_card2return_h(self, cards_in_hand):
        # Have the human player to return a card to the donor
        card2return = int(input('\n\U0001F381 Please input a card you wish to return, rank_card={}\U0001F4AA: '.format(
            rank2str[self.rank2play])))
        
        while not isinstance(card2return, int)\
            or card2return not in cards_in_hand\
            or card2rank(card2return) > 10:
            card2return = int(input('\nInvalid input! Please input a card you wish to return, rank_card={}\U0001F4AA: '.format(
                rank2str[self.rank2play])))
        
        return card2return


    def get_card2donate(self, cards_in_hand):
        player_rankAndCards = {}
        player_wild_cards = []

        for c in cards_in_hand:
            c_rank = card2rank(c)
            if c_rank == self.rank2play and card2suit(c) % 4 == 0:
                player_wild_cards.append(c)
            else:  # exclude wild cards
                if c_rank in player_rankAndCards:
                    player_rankAndCards[c_rank].append(c)
                else:
                    player_rankAndCards[c_rank] = [c]
        
        card2donate_cands = player_rankAndCards[max(player_rankAndCards.keys())]
        player_mg = MovesGenerator(cards_in_hand)
        player_straight_flush = player_mg.gen_straight_flush()
        if len(player_wild_cards) > 0:
            player_straight_flush += player_mg.gen_cards_straight_filtered_w(player_wild_cards, -1)

        card2donate_cands_copy = card2donate_cands.copy()
        for player_sf in player_straight_flush:
            for c2d in card2donate_cands_copy:
                if c2d in player_sf:
                    card2donate_cands_copy.remove(c2d)
        
        if len(card2donate_cands_copy) > 0:
            return random.choice(card2donate_cands_copy)

        return random.choice(card2donate_cands)


    def step_h(self, role='None', display_cards=False):        
        if self.num_steps == 0 and len(self.prev_finishingOrder) == 4 \
            and max(self.playerAndRank.values()) > 2:  # process tribute
            pf1 = self.prev_finishingOrder[0]
            pf2 = self.prev_finishingOrder[1]
            pf3 = self.prev_finishingOrder[2]
            pf4 = self.prev_finishingOrder[3]
            
            pf1_cards = self.info_sets[playerNum2Str[pf1]].player_hand_cards
            pf2_cards = self.info_sets[playerNum2Str[pf2]].player_hand_cards
            pf3_cards = self.info_sets[playerNum2Str[pf3]].player_hand_cards
            pf4_cards = self.info_sets[playerNum2Str[pf4]].player_hand_cards
            pf34_cards = pf3_cards + pf4_cards

            double_donors = abs(pf4 - pf3) == 2
            if double_donors and 54 in pf34_cards and 108 in pf34_cards \
                or not double_donors and 54 in pf4_cards and 108 in pf4_cards:
                if display_cards:
                    print("\u2757 Tribute defied, previous winner to play first..")

                # skip to pf1 as the first player (instead of pf4) when no tribute is to be made..
                self.turns2skip = steps2player[pf1][pf4]
            else:  # When the tribute process MUST be carried out..
                card2return_pf1 = self.get_card2return(pf1_cards)
                card2donate_pf4 = self.get_card2donate(pf4_cards)

                # Have the human player to return a card to the donor
                if role == playerNum2Str[pf1]:
                    self.display_cards_in_hand(role)
                    card2return_pf1 = self.get_card2return_h(pf1_cards)
                            
                if double_donors:
                    card2return_pf2 = self.get_card2return(pf2_cards)
                    
                    # Have the human player to return a card to the donor
                    if role == playerNum2Str[pf2]:
                        self.display_cards_in_hand(role)
                        card2return_pf2 = self.get_card2return_h(pf2_cards)

                    card2donate_pf3 = self.get_card2donate(pf3_cards)

                    card2donate_rank_pf3 = card2rank(card2donate_pf3)
                    card2donate_rank_pf4 = card2rank(card2donate_pf4)

                    pf1234 = pf1 * 1000 + pf2 * 100 + pf3 * 10 + pf4

                    if card2donate_rank_pf3 > card2donate_rank_pf4\
                         or card2donate_rank_pf3 == card2donate_rank_pf4 and\
                            (pf1234 == 1324 or pf1234 == 3142 or pf1234 == 2431 or pf1234 == 4213):
                        pf1_cards.append(card2donate_pf3)
                        pf3_cards.remove(card2donate_pf3)

                        pf1_cards.remove(card2return_pf1)
                        pf3_cards.append(card2return_pf1)
                        
                        pf2_cards.append(card2donate_pf4)
                        pf4_cards.remove(card2donate_pf4)

                        pf2_cards.remove(card2return_pf2)
                        pf4_cards.append(card2return_pf2)

                        if display_cards:
                            self.display_tribute_info(playerNum2Str[pf3], playerNum2Str[pf1], card2donate_pf3, card2return_pf1)
                            self.display_tribute_info(playerNum2Str[pf4], playerNum2Str[pf2], card2donate_pf4, card2return_pf2)
                        
                        self.turns2skip = 2
                    else:  # "default": pf4 <> pf1; pf3 <> pf2
                        pf1_cards.append(card2donate_pf4)
                        pf4_cards.remove(card2donate_pf4)

                        pf1_cards.remove(card2return_pf1)
                        pf4_cards.append(card2return_pf1)
                        
                        pf2_cards.append(card2donate_pf3)
                        pf3_cards.remove(card2donate_pf3)

                        pf2_cards.remove(card2return_pf2)
                        pf3_cards.append(card2return_pf2)

                        if display_cards:
                            self.display_tribute_info(playerNum2Str[pf4], playerNum2Str[pf1], card2donate_pf4, card2return_pf1)
                            self.display_tribute_info(playerNum2Str[pf3], playerNum2Str[pf2], card2donate_pf3, card2return_pf2)
                            
                else:  # single donor
                    pf1_cards.append(card2donate_pf4)
                    pf4_cards.remove(card2donate_pf4)

                    pf1_cards.remove(card2return_pf1)
                    pf4_cards.append(card2return_pf1)
                    
                    if display_cards:
                        self.display_tribute_info(playerNum2Str[pf4], playerNum2Str[pf1], card2donate_pf4, card2return_pf1)
        ### END of processing tribute

        ### Process turn-skipping as a result of tribute defiance/double donors
        if self.num_steps < self.turns2skip:
            self.num_steps += 1
            action = []
            self.last_move_dict[self.acting_player_position] = action.copy()

            self.get_acting_player_position()  # set next player
            self.game_infoset = self.get_infoset()
            return
        
        self.num_steps += 1
        no_yield = True
        
        # Force the blocking non-partner player to yield if winner 1/2 is decided recently AND no contests happened in between
        if self.w1_step != -1 and self.num_steps == self.w1_step + 5 and self.w1_yield \
            or self.w2_step != -1 \
                and len(self.finishingOrder) >= 2 \
                and abs(self.finishingOrder[0] - self.finishingOrder[1]) != 2 \
                and self.acting_player_position not in self.finishingOrder \
                and self.num_steps == self.w2_step + 5 and self.w2_yield:
            action = []
            no_yield = False

        if no_yield:
            if self.acting_player_position == role:
                # if len(self.info_sets[role].player_hand_cards) < 10:
                #     print('legal actions: {}'.format(self.get_legal_card_play_actions()))

                ### Display remaining cards in each player's hand ###
                print('\nRemaining cards \u1367\u1367', end='')
                if self.info_sets[self.acting_player_position].num_cards_left_dict == None:
                    print(self.info_sets[self.acting_player_position].num_cards_left_dict)
                else:
                    for p in ['p1', 'p2', 'p3', 'p4']:
                        if p == self.acting_player_position:
                            print(p + '(you): {} \u1367\u1367'.format(self.info_sets[self.acting_player_position].num_cards_left_dict[p]), end='')
                        else:
                            print(p + ': {} \u1367\u1367'.format(self.info_sets[self.acting_player_position].num_cards_left_dict[p]), end='')
                            
                    print()
                
                if len(self.info_sets[role].player_hand_cards) > 0:
                    self.display_cards_in_hand(role)
                    
                    human_input = input('\nPlease input card(s) you wish to play, rank_card={}\U0001F4AA: '.format(
                        rank2str[self.rank2play]))
                    if human_input == 'all':
                        action = self.info_sets[role].player_hand_cards.copy()
                    else:
                        action = self.str2action(human_input)

                    if len(action) > 0:
                        while not all(e in self.info_sets[role].player_hand_cards for e in action) \
                            or action not in self.get_legal_card_play_actions():
                            action = self.str2action(input('Invalid input! Please input again: '))
                elif len(self.info_sets[role].player_hand_cards) == 0:
                    action = []
                    print('You have won, please wait for other players...')

                print('-' * separator_len)
            else:  # as AI
                # auto passes if there is no more cars in hand
                if len(self.info_sets[self.acting_player_position].player_hand_cards) == 0:
                    action = []
                else:
                    action = self.players[self.acting_player_position].act(self.game_infoset)
                    assert action in self.game_infoset.legal_actions

        len_action = len(action)
        
        coop_state = 0
        if self.num_steps >= 3:
            if abs(playerStr2Num[self.acting_player_position] - playerStr2Num[self.last_pid]) == 2:
                if len(self.game_infoset.legal_actions) >= 2:
                    coop_state = 1
                    self.coop_counter[self.acting_player_position][1] += 1  # could've crushed
                    if len_action > 0:
                        coop_state = 2
                        self.coop_counter[self.acting_player_position][0] += 1  # actually crushed

        self.coop_status[self.acting_player_position][0] = coop_state

        # print('active player: {}|last_pid: {}'.format(self.acting_player_position, self.last_pid))
        opponent_min_cards = DECK_SIZE
        partner_min_cards = DECK_SIZE
        
        if self.acting_player_position == self.last_pid:  # if the acting_player leads to play    
            for p in ['p1', 'p2', 'p3', 'p4']:
                if p != self.acting_player_position:
                    p_cards_left = self.info_sets[self.acting_player_position].num_cards_left_dict[p]
                    if abs(playerStr2Num[p] - playerStr2Num[self.acting_player_position]) == 2:
                        if  p_cards_left <= 5:
                            partner_min_cards = p_cards_left
                    else:
                        if p_cards_left <= 5 and p_cards_left < opponent_min_cards:
                            opponent_min_cards = p_cards_left

        coop_state = 3
        if opponent_min_cards <= 5:
            for la in self.game_infoset.legal_actions:
                if len(la) > opponent_min_cards:
                    coop_state = 4  # could've dwarved the opponent
                    self.coop_counter[self.acting_player_position][3] += 1
                    break
            
            if len(action) > opponent_min_cards:
                coop_state = 5  # actually dwarved the opponent
                self.coop_counter[self.acting_player_position][2] += 1

        self.coop_status[self.acting_player_position][1] = coop_state 
        
        coop_state = 6
        if partner_min_cards <= 5:
            for la in self.game_infoset.legal_actions:
                if 0 < len(la) <= opponent_min_cards:
                    coop_state = 7  # could've assisted partner
                    self.coop_counter[self.acting_player_position][5] += 1 
                    break
            
            if 0 < len(action) <= partner_min_cards:
                coop_state = 8  # actually assisted partner
                self.coop_counter[self.acting_player_position][4] += 1

        self.coop_status[self.acting_player_position][2] = coop_state

        # Any contests?
        if len_action > 0:
            if self.w1_step != -1 \
                and self.w1_step + 1 <= self.num_steps <= self.w1_step + 3:
                if self.w1_yield:
                    self.w1_yield = False

            if self.w2_step != -1 \
                and self.w2_step + 1 <= self.num_steps <= self.w2_step + 3:
                if self.w2_yield:
                    self.w2_yield = False

            self.last_pid = self.acting_player_position
            self.is_new_round = 0
        elif len_action == 0:
            self.is_new_round += 1
            
        if display_cards:
            if role not in ['p1', 'p2', 'p3', 'p4'] and \
                self.is_new_round == 2 and self.last_pid != None:
                print('-' * separator_len)
                print('<<< Cards in hands >>>\np1: {}\np2: {}\np3: {}\np4: {}'.format(
                    self.info_sets['p1'].player_hand_cards,
                    self.info_sets['p2'].player_hand_cards,
                    self.info_sets['p3'].player_hand_cards,
                    self.info_sets['p4'].player_hand_cards,
                ))
                print('-' * separator_len)

            ### Display cards being played ###
            print('\u23E9' + self.acting_player_position, end=' ')
            if len_action > 0:
                print('plays |', end='')
                for a in action:
                    print(a, end='|')
                
                print('\u2263\u2263|', end='')
                a_rank, a_symb = [], []
                for a in action:
                    if a == 53 or a == 107:
                        a_rank.append(20)
                        a_symb.append('\033[31m鬼\033[0m')
                    elif a == 54 or a == 108:
                        a_rank.append(30)
                        a_symb.append('鬼')
                    else:
                        a_rank.append(card2rank(a))
                        a_symb.append(row2symbol[card2row[a]])

                ### rearrange for `A`
                a_sorted_ind = np.argsort(a_rank)
                a_rank_sorted = [a_rank[a_sorted_ind[a]] for a in range(len_action)]
                a_symb_sorted = [a_symb[a_sorted_ind[a]] for a in range(len_action)]

                move_dict = collections.Counter(a_rank_sorted)
                if a_rank_sorted[0] == 2 and a_rank_sorted[-1] == 14:
                    cutoff = 0
                    if len_action == 5:
                        if len(move_dict) == 5:
                            cutoff = -1
                    elif len_action == 6:
                        if len(move_dict) == 3:
                            cutoff = -2
                        elif len(move_dict) == 2:
                            cutoff = -3

                    if -3 <= cutoff < 0:
                        a_rank_sorted = a_rank_sorted[cutoff:] + a_rank_sorted[:cutoff]
                        a_symb_sorted = a_symb_sorted[cutoff:] + a_symb_sorted[:cutoff]

                for a in range(len_action):   
                    print(rank2str[a_rank_sorted[a]] + a_symb_sorted[a], end='|')
            else:  # if the action is to pass
                if no_yield:
                    print('passes', end='')
                else:
                    print('passes, yielding to recent winner\'s partner\u2757', end='')
                
            print()

        self.last_move_dict[self.acting_player_position] = action.copy()

        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)
        
        self.played_cards[self.acting_player_position] += action

        self.game_done()  # onGameEnd()

        if not self.game_over:  # rotate to the next player
            self.get_acting_player_position()  # set next player
            self.game_infoset = self.get_infoset()  # update infoset
            # self.get_acting_player_position()  # set next player


    def get_last_move(self):
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                if len(self.card_play_action_seq) == 1:
                    last_move = [[]]
                else:
                    if len(self.card_play_action_seq[-2]) == 0:
                        last_move = self.card_play_action_seq[-3]
                    else:
                        last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]

        return last_move


    def get_last_three_moves(self):
        last_three_moves = [[], [], []]
        for card in self.card_play_action_seq[-3:]:
            last_three_moves.insert(0, card)
            last_three_moves = last_three_moves[:3]
        return last_three_moves


    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'p1'
        else:
            if self.acting_player_position == 'p1':
                self.acting_player_position = 'p2'

            elif self.acting_player_position == 'p2':
                self.acting_player_position = 'p3'

            elif self.acting_player_position == 'p3':
                self.acting_player_position = 'p4'

            else:
                self.acting_player_position = 'p1'

        return self.acting_player_position


    def update_acting_player_hand_cards(self, action):
        if action != []:
            for card in action:
                if card in self.info_sets[self.acting_player_position].player_hand_cards:
                    self.info_sets[self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()


    def get_acting_players_wild_cards(self):
        result = []
        for card in self.info_sets[self.acting_player_position].player_hand_cards:
            if card2rank(card) == self.rank2play and card2suit(card) % 4 == 0:
                result.append(card)
        
        return result


    def get_legal_card_play_actions(self):
        mg = MovesGenerator(
            self.info_sets[self.acting_player_position].player_hand_cards)

        action_sequence = self.card_play_action_seq

        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                if len(action_sequence) == 1:
                    rival_move = [[]]
                else:    
                    if len(action_sequence[-2]) == 0:
                        rival_move = action_sequence[-3]
                    else:
                        rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move, rank2play=self.rank2play)
        rival_move_type = rival_type['type']
        rival_move_rank = rival_type.get('rank', -1)

        # rival_move_len = rival_type.get('len', 1)
        # print('rival move info, type: {}, rank: {}'.format(rival_move_type, rival_move_rank))

        moves = list()
        wild_cards_ = self.get_acting_players_wild_cards()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()
            if len(wild_cards_) > 0:
                for reps_ in range(2, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank, reps_)
                
                moves += mg.gen_cards_straight_filtered_w(wild_cards_, -1)
                moves += mg.gen_cards_straight_flush_filtered_w(wild_cards_, -1)
                moves += mg.gen_cards_steel_plate_filtered_w(wild_cards_, -1)
                moves += mg.gen_cards_wooden_plate_filtered_w(wild_cards_, -1)
                moves += mg.gen_type_7_3_2_filtered_w(wild_cards_, -1)

        elif rival_move_type == md.TYPE_1_SINGLE:
            moves = ms.filter_type_1_single(mg.gen_cards_reps(1), rival_move, self.rank2play)

        elif rival_move_type == md.TYPE_2_PAIR:
            moves = ms.filter_type_2_pair(mg.gen_cards_reps(2), rival_move, rival_move_rank, self.rank2play)
            if len(wild_cards_) > 0:
               moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank, 2)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            moves = ms.filter_type_3_triple(mg.gen_cards_reps(3), rival_move, rival_move_rank, self.rank2play)
            if len(wild_cards_) > 0:
                moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank, 3)

        elif rival_move_type == md.TYPE_4_BOMB:                
            moves = ms.filter_type_4_bomb(mg.gen_cards_reps(4), rival_move, rival_move_rank, self.rank2play) \
                + mg.gen_straight_flush() \
                + mg.gen_cards_reps(5) + mg.gen_cards_reps(6) \
                + mg.gen_cards_reps(7) + mg.gen_cards_reps(8) \
                + mg.gen_type_5_king_bomb()
            
            if len(wild_cards_) > 0:
                for reps_ in range(4, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank if reps_ == 4 else -1, reps_)

                moves += mg.gen_cards_straight_flush_filtered_w(wild_cards_, -1)

        elif rival_move_type == md.TYPE_45_BOMB:
            moves = ms.filter_type_45_bomb(mg.gen_cards_reps(5), rival_move, rival_move_rank, self.rank2play) \
                + mg.gen_straight_flush() \
                + mg.gen_cards_reps(6) \
                + mg.gen_cards_reps(7) + mg.gen_cards_reps(8) \
                + mg.gen_type_5_king_bomb()

            if len(wild_cards_) > 0:
                for reps_ in range(5, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank if reps_ == 5 else -1, reps_)

                moves += mg.gen_cards_straight_flush_filtered_w(wild_cards_, -1)

        elif rival_move_type == md.TYPE_6_STRAIGHT_FLUSH:
            moves = ms.filter_type_6_straight_flush(mg.gen_straight_flush(), rival_move, rival_move_rank) \
                + mg.gen_cards_reps(6) \
                + mg.gen_cards_reps(7) + mg.gen_cards_reps(8) \
                + mg.gen_type_5_king_bomb()
            
            if len(wild_cards_) > 0:
                for reps_ in range(6, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, -1, reps_)
                
                moves += mg.gen_cards_straight_flush_filtered_w(wild_cards_, rival_move_rank)

        elif rival_move_type == md.TYPE_46_BOMB:
            moves = ms.filter_type_46_bomb(mg.gen_cards_reps(6), rival_move, rival_move_rank, self.rank2play) \
                + mg.gen_cards_reps(7) + mg.gen_cards_reps(8) \
                + mg.gen_type_5_king_bomb()
            
            if len(wild_cards_) > 0:
                for reps_ in range(6, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank if reps_ == 6 else -1, reps_)

        elif rival_move_type == md.TYPE_47_BOMB:
            moves = ms.filter_type_47_bomb(mg.gen_cards_reps(7), rival_move, rival_move_rank, self.rank2play) \
                + mg.gen_cards_reps(8) \
                + mg.gen_type_5_king_bomb()
            
            if len(wild_cards_) > 0:
                for reps_ in range(7, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank if reps_ == 7 else -1, reps_)
        
        elif rival_move_type == md.TYPE_48_BOMB:
            moves = ms.filter_type_48_bomb(mg.gen_cards_reps(8), rival_move, rival_move_rank, self.rank2play) \
                + mg.gen_type_5_king_bomb()
            
            if len(wild_cards_) > 0:
                moves += mg.gen_cards_reps_filtered_w(wild_cards_, rival_move_rank, 8)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == md.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move_rank, self.rank2play)
            if len(wild_cards_) > 0:
                moves += mg.gen_type_7_3_2_filtered_w(wild_cards_, rival_move_rank)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg._gen_serial_moves(5, 1)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move, rival_move_rank)
            
            if len(wild_cards_) > 0:
                moves += mg.gen_cards_straight_filtered_w(wild_cards_, rival_move_rank)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg._gen_serial_moves(3, 2)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move, rival_move_rank)

            if len(wild_cards_) > 0:
                moves += mg.gen_cards_wooden_plate_filtered_w(wild_cards_, rival_move_rank)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg._gen_serial_moves(2, 3)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move, rival_move_rank)
            
            if len(wild_cards_) > 0:
                moves += mg.gen_cards_steel_plate_filtered_w(wild_cards_, rival_move_rank)

        if rival_move_type not in [md.TYPE_0_PASS,
                                   md.TYPE_4_BOMB,
                                   md.TYPE_45_BOMB, md.TYPE_46_BOMB,
                                   md.TYPE_47_BOMB, md.TYPE_48_BOMB,
                                   md.TYPE_6_STRAIGHT_FLUSH,
                                   md.TYPE_5_KING_BOMB]:
            moves = moves + mg.gen_cards_reps(4) \
                    + mg.gen_cards_reps(5) + mg.gen_cards_reps(6) \
                    + mg.gen_cards_reps(7) + mg.gen_cards_reps(8) \
                    + mg.gen_straight_flush() \
                    + mg.gen_type_5_king_bomb()
            
            if len(wild_cards_) > 0:
                for reps_ in range(4, 9):
                    moves += mg.gen_cards_reps_filtered_w(wild_cards_, -1, reps_)
                
                moves += mg.gen_cards_straight_flush_filtered_w(wild_cards_, -1)

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            # if isinstance(m, int):
            #     print(m)
            m.sort()

        if len(moves) == 0:  # 'pass' IS a legal move!
            moves = moves + [[]]

        return moves


    def reset(self):
        self.card_play_action_seq = []

        # self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        # self.player_utility_dict = None
        self.player_utility_dict = {}

        self.last_move_dict = {'p1': [],
                               'p2': [],
                               'p3': [],
                               'p4': []}

        self.played_cards = {'p1': [],
                             'p2': [],
                             'p3': [],
                             'p4': []}

        self.last_move = []
        self.last_three_moves = []

        self.info_sets = {'p1': InfoSet('p1'),
                         'p2': InfoSet('p2'),
                         'p3': InfoSet('p3'),
                         'p4': InfoSet('p4'),}

        self.last_pid = playerNum2Str[self.first_player]
        self.finishingOrder = []
        

        self.num_steps = 0
        self.w1_step = -1
        self.w2_step = -1
        self.w1_yield = True
        self.w2_yield = True
        # self.rank2play = 2  # ⚠ do NOT set this or self.rank2play would not get carried over to the next round!
        # self.prev_finishingOrder = []

        self.coop_status = { 'p1': [0, 0, 0], 'p2': [0, 0, 0], 'p3': [0, 0, 0], 'p4': [0, 0, 0] }
        self.turns2skip = -1
    ### END of `reset()`


    def get_infoset(self):
        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        self.info_sets[
            self.acting_player_position].legal_actions = self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].rank2play = self.rank2play

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_three_moves = self.get_last_three_moves()

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in ['p1', 'p2', 'p3', 'p4']}

        self.info_sets[self.acting_player_position].other_hand_cards = []
        for pos in ['p1', 'p2', 'p3', 'p4']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards
        
        # self.info_sets[self.acting_player_position].three_landlord_cards = \
        #     self.three_landlord_cards  # tribute # placeholder
        
        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in ['p1', 'p2', 'p3', 'p4']}

        self.info_sets[
             self.acting_player_position].coop_status = self.coop_status.copy()

        return deepcopy(self.info_sets[self.acting_player_position])


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """
    def __init__(self, player_position):
        # The player position, i.e., p1, p2, p3, p4
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int 
        self.num_cards_left_dict = None
        
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player 
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_three_moves = None
        # The last moves for all the postions
        self.last_move_dict = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict. 
        self.all_handcards = None

        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None

        self.rank2play = None

        self.coop_status = None

        # tribute (the type is list) # placeholder
        # self.three_landlord_cards = None
        
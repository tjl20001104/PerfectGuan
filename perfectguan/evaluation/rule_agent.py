from ..env.utils import card2col, card2row, card2rank, card2suit
from ..env import move_detector as md
from ..env.stepwise_reward import action2specific

from .rule_AI_2.PlayCard import PlayCard
from .rule_AI_2.strategy import Strategy0
from .rule_AI_1.action import Action

import numpy as np
import collections


type2str = {0:'H', 1:'D', 2:'C', 3:'S', 4:'H', 5:'D', 6:'C', 7:'S'}
rank2str = {1:'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
            11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17:'B', 20: 'B', 30: 'R'}
idx2actiontype = {0:'PASS', 1:'Single', 2:'Pair', 3:'Trips', 4:'Bomb',
                  5:'Bomb', 6:'StraightFlush', 7:'ThreeWithTwo', 8:'Straight',
                  9:'ThreePair', 10:'TwoTrips', 45:'Bomb', 46:'Bomb',
                  47:'Bomb', 48:'Bomb'}


class RuleAgent_1():
    def __init__(self):
        self.name = 'Rule'
        self.action = Action()
        self.pass_num = 0
        self.my_pass_num = 0
        self.row2type = {0:'H', 1:'D', 2:'C', 3:'S',
                         4:'H', 5:'D', 6:'C', 7:'S',
                         8:'B', 9:'R'}

    def act(self, infoset):
        # if can only pass, pass
        num_c = infoset.num_cards_left_dict
        num_of_player_cards = [num_c['p1'],num_c['p2'],num_c['p3'],num_c['p4']]
        my_pos = (int(infoset.player_position[1]) - 1) % 4
        if infoset.legal_actions == [[]] and num_of_player_cards[my_pos] != 0:
            self.my_pass_num += 1
            self.pass_num += 1
            return []
        
        message = self.info2message(infoset)
        remain_cards = self.generate_remain_cards(infoset)

        # play first or play positively
        if [] not in infoset.legal_actions:
            self.my_pass_num = 0
            self.pass_num = 0
            ActionIdx = self.action.active(
                message['actionList'],
                message['handCards'],
                message['curRank'],
                num_of_player_cards,
                my_pos,
                remain_cards
                )

        # play passively
        else:
            if len(infoset.card_play_action_seq) > 2:
                if infoset.card_play_action_seq[-2] == [] and num_of_player_cards[(my_pos+2)%4] != 0:
                    self.pass_num += 1
                else:
                    self.pass_num = 0
            ActionIdx = self.action.passive(
                message['actionList'],
                message['handCards'],
                message['curRank'],
                message['curAction'],
                message['greaterAction'],
                my_pos,
                message['greaterPos'],
                remain_cards,
                num_of_player_cards,
                self.pass_num,
                self.my_pass_num
            )

        # return a list of card id representing action like [50,64,65,66,90,92]
        action = infoset.legal_actions[ActionIdx]
        if action == [] and num_of_player_cards[my_pos] != 0:
            self.my_pass_num += 1
            self.pass_num += 1
        else:
            self.my_pass_num = 0
            self.pass_num = 0
        return action
    
    def info2message(self, infoset):
        message = {}
        rank = infoset.rank2play
        actionList = [self.action2list(a, rank) for a in infoset.legal_actions]
        message['actionList'] = actionList
        message['handCards'] = self.cards2str(infoset.player_hand_cards)
        message['curRank'] = rank2str[rank]
        if len(infoset.card_play_action_seq) == 0:
            message['curAction'] = -1
            message['greaterAction'] = None
            message['greaterPos'] = -1
        else:
            message['curAction'] = self.action2list(infoset.card_play_action_seq[-1], rank)
            message['greaterAction'] = self.action2list(infoset.last_move, rank)
            message['greaterPos'] = (int(infoset.last_pid[1]) - 1) % 4
        return message
        
    def cards2str(self, cards):
        len_cards = len(cards)
        result = []
        rank,symb = [],[]
        for card in cards:
            if card == 53 or card == 107:
                rank.append(20)
                symb.append(3)
            elif card == 54 or card == 108:
                rank.append(30)
                symb.append(0)
            else:
                rank.append(card2rank(card))
                symb.append(card2row[card])
        sorted_ind = np.argsort(rank)
        rank_sorted = [rank[sorted_ind[a]] for a in range(len_cards)]
        symb_sorted = [symb[sorted_ind[a]] for a in range(len_cards)]

        move_dict = collections.Counter(rank_sorted)
        if rank_sorted[0] == 2 and rank_sorted[-1] == 14:
            cutoff = 0
            if len_cards == 5:
                if len(move_dict) == 5:
                    cutoff = -1
            elif len_cards == 6:
                if len(move_dict) == 3:
                    cutoff = -2
                elif len(move_dict) == 2:
                    cutoff = -3

            if -3 <= cutoff < 0:
                rank_sorted = rank_sorted[cutoff:] + rank_sorted[:cutoff]
                symb_sorted = symb_sorted[cutoff:] + symb_sorted[:cutoff]
        for a in range(len_cards):
            result.append(type2str[symb_sorted[a]]+rank2str[rank_sorted[a]])
        return result
    
    def action2list(self, action, rank):
        if action == []:
            return ['PASS','PASS','PASS']
        cards = self.cards2str(action)
        move_type = md.get_move_type(action, rank)
        action_type = move_type['type']
        action_type = idx2actiontype[action_type]
        action_rank = move_type.get('rank', -1)
        if action_rank > -1:
            action_rank = rank2str[action_rank]
        return [action_type,action_rank,cards]
    
    def generate_remain_cards(self, infoset):
        remain_cards = infoset.player_hand_cards + infoset.other_hand_cards
        result = {
            "S": [0] * 14,  # s黑桃
            "H": [0] * 14,  # h红桃
            "C": [0] * 14,  # c方块
            "D": [0] * 14,  # d梅花
        }
        for card in remain_cards:
            if card == 53 or card == 107:
                m_rank = 13
                m_suit = 'S'
            elif card == 54 or card == 108:
                m_rank = 13
                m_suit = 'H'
            else:
                m_rank = card2col[card]
                m_suit = type2str[card2row[card]]
            result[m_suit][m_rank] += 1
        return result


class RuleAgent_2():
    def __init__(self):
        self.name = 'Rule'
        self.row2type = {0:'H', 1:'D', 2:'C', 3:'S',
                         4:'H', 5:'D', 6:'C', 7:'S',
                         8:'B', 9:'R'}

    def act(self, infoset, return_idx=False):
        if infoset.legal_actions == [[]]:
            if return_idx == True:
                return 366
            return []
        
        message = self.info2message(infoset)
        rank = infoset.rank2play
        my_pos = (int(infoset.player_position[1])-1) % 4
        last_pid = infoset.last_pid
        num_c = infoset.num_cards_left_dict
        num_of_player_cards = [num_c['p1'],num_c['p2'],num_c['p3'],num_c['p4']]
        remain_cards = self.generate_remain_cards(infoset)
        action_history = [self.action2list(a, rank) for a in infoset.card_play_action_seq]
        
        if len(infoset.card_play_action_seq) < 4:
            Strategy0.SetBeginning(
                my_pos,
                message['handCards']
            )
        else:
            Strategy0.UpdatePlay(
                my_pos,
                message['curAction'],
                message['greaterPos'],
                message['greaterAction'],
                num_of_player_cards,
                remain_cards,
                action_history
            )

        Strategy0.UpdateCurRank(
            message['curRank']
        )

        # play first or play positively
        if [] not in infoset.legal_actions:
            retValue = PlayCard().FreePlay(
                message['handCards'],
                message['curRank'],
                message['actionList']
            )

        # play passively
        else:
            a = message['greaterAction']
            formerAction = {'type':a[0], 'rank':a[1], 'action':a[2]}
            retValue = PlayCard().RestrictedPlay(
                message['handCards'],
                formerAction,
                message['curRank'],
                message['actionList']
            )

        sortedAction = retValue["action"]
        if retValue["type"] != "PASS":
            sortedAction.sort()
        retIndex = 0
        for action in message["actionList"]:
            if (action[2]!="PASS"): action[2].sort()
            #print("retvalue:",retValue["type"], retValue["rank"], sortedAction)
            #print("actionfromlist:",action[0], action[1], action[2])
            if (action[0]==retValue["type"] and action[1]==retValue["rank"] and action[2]==sortedAction):
                retIndex=message["actionList"].index(action)

        action = infoset.legal_actions[retIndex]
        if return_idx == True:
            if return_idx == []:
                return 366
            else:
                _,a_id = action2specific(action, infoset.rank2play)
                return a_id
        return action
    
    def info2message(self, infoset):
        message = {}
        rank = infoset.rank2play
        actionList = [self.action2list(a, rank) for a in infoset.legal_actions]
        message['actionList'] = actionList
        message['handCards'] = self.cards2str(infoset.player_hand_cards)
        message['curRank'] = rank2str[rank]
        if len(infoset.card_play_action_seq) == 0:
            message['curAction'] = -1
            message['greaterAction'] = None
            message['greaterPos'] = -1
        else:
            message['curAction'] = self.action2list(infoset.card_play_action_seq[-1], rank)
            message['greaterAction'] = self.action2list(infoset.last_move, rank)
            message['greaterPos'] = (int(infoset.last_pid[1]) - 1) % 4
        return message
        
    def cards2str(self, cards):
        len_cards = len(cards)
        result = []
        rank,symb = [],[]
        for card in cards:
            if card == 53 or card == 107:
                rank.append(20)
                symb.append(3)
            elif card == 54 or card == 108:
                rank.append(30)
                symb.append(0)
            else:
                rank.append(card2rank(card))
                symb.append(card2row[card])
        sorted_ind = np.argsort(rank)
        rank_sorted = [rank[sorted_ind[a]] for a in range(len_cards)]
        symb_sorted = [symb[sorted_ind[a]] for a in range(len_cards)]

        move_dict = collections.Counter(rank_sorted)
        if rank_sorted[0] == 2 and rank_sorted[-1] == 14:
            cutoff = 0
            if len_cards == 5:
                if len(move_dict) == 5:
                    cutoff = -1
            elif len_cards == 6:
                if len(move_dict) == 3:
                    cutoff = -2
                elif len(move_dict) == 2:
                    cutoff = -3

            if -3 <= cutoff < 0:
                rank_sorted = rank_sorted[cutoff:] + rank_sorted[:cutoff]
                symb_sorted = symb_sorted[cutoff:] + symb_sorted[:cutoff]
        for a in range(len_cards):
            result.append(type2str[symb_sorted[a]]+rank2str[rank_sorted[a]])
        return result
    
    def action2list(self, action, rank):
        if len(action) == 0:
            return ['PASS','PASS','PASS']
        cards = self.cards2str(action)
        move_type = md.get_move_type(action, rank)
        action_type = move_type['type']
        action_type = idx2actiontype[action_type]
        action_rank = move_type.get('rank', -1)
        if action_rank > -1:
            action_rank = rank2str[action_rank]
        if move_type['type'] == 5:
            action_rank = 'JOKER'
        return [action_type,action_rank,cards]
    
    def generate_remain_cards(self, infoset):
        remain_cards = infoset.other_hand_cards
        result = {a:0 for a in ['2','3','4','5','6','7','8','9','T','J','Q','K','A','B','R']}
        for card in remain_cards:
            rank = rank2str[card2rank(card)]
            result[rank] += 1
        return result
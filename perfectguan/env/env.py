from collections import Counter
import numpy as np

from perfectguan.env.game import GameEnv
from perfectguan.env.utils import card2col, card2row
from perfectguan.config import DECK_SIZE, MAX_ACTION, ACTION_SIZE
from perfectguan.env.stepwise_reward import min_steps_play_out, actions_merge

deck = list(range(1, DECK_SIZE + 1)) # using 2 decks: 54 x 2

class Env:
    def __init__(self, objective):
        """
        Objective is wp. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        self.players = {}
        for position in ['p1', 'p2', 'p3', 'p4']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        # Initialize the internal environment
        self._env = GameEnv(self.players)  # env from game.py

        self.infoset = None
        self.first_player = self._env.first_player
        self.player_utility_dict = self._env.player_utility_dict

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'p1': _deck[:27],
                          'p2': _deck[27:54],
                          'p3': _deck[54:81],
                          'p4': _deck[81:]
                          # 'three_landlord_cards': _deck[17:20]  # tribute # placeholder
                          }
        
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step_h()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        # reward = None
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)

        return obs, reward, done, self._env.player_utility_dict

    def _get_reward(self):  # 1/-1 or more?
        """
        This function is called at the end of each
        game. It returns 1/-1 for win/loss,
        """
        winner = self._env.finishingOrder[0]
        
        if winner == 1 or winner == 3:
            return 1.0
        else:
            return -1.0


    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_rank2play(self):
        """
        The current rank to play. This is used as
        a feature of the neural network.
        """
        return self._env.get_rank2play()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be p* for * in [1,2,3,4] 
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        return self._env.game_over
    
    
class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.
    
    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be p* for * in [1,2,3,4]

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `z`: same as z_batch but not a batch.

    `legal_actions` is the legal moves

    `x_no_action`: other features exluding the hitorical moves and
    the action features. It does not have the batch dim.
    """

    if infoset.player_position == 'p1':
        return _get_obs_p1(infoset)
    elif infoset.player_position == 'p2':
        return _get_obs_p2(infoset)
    elif infoset.player_position == 'p3':
        return _get_obs_p3(infoset)
    elif infoset.player_position == 'p4':
        return _get_obs_p4(infoset)
    else:
        raise ValueError('')


def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    if type(num_left_cards) is not int:
        num_left_cards = int(num_left_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot


def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(DECK_SIZE, dtype=np.int8)

    matrix = np.zeros([8, 13], dtype=np.int8)
    jokers = np.zeros(4, dtype=np.int8)

    for card in list_cards:
        if card <= 52 or 55 <= card <= 106:
            matrix[card2row[card], card2col[card]] = 1
        elif card == 53:
            jokers[0] = 1
        elif card == 54:
            jokers[1] = 1
        elif card == 107:
            jokers[2] = 1
        elif card == 108:
            jokers[3] = 1

    return np.concatenate((matrix[:4,:].flatten('F'), jokers[:2], 
                           matrix[4:,:].flatten('F'), jokers[2:]))  # of shape (108,)


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 -> 20 actions. If there is
    not enough actions, we pad the features with 0's. Since
    three -> four moves is a round in DouDizhu -> GuanDan, 
    we concatenate the representations for each consecutive 
    three -> four moves. Finally, we obtain a 5x162 -> 5x416 matrix, 
    which will be fed into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), DECK_SIZE))  # was: 54
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 4 * DECK_SIZE)  # was: 5, 162
    return action_seq_array

def _process_action_seq(sequence, length=20):
    """
    A utility function encoding historical moves. We
    encode 15 -> 20 moves. If there are not enough moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _get_one_hot_rank2play(rank2play):
    """
    A utility function to encode rank card
    into one-hot representation.
    """
    one_hot = np.zeros(13)
    one_hot[rank2play - 2] = 1
    return one_hot


def _get_obs_p1(infoset):
    legal_actions = infoset.legal_actions
    my_handcards = _cards2array(infoset.player_hand_cards)
    
    p2_handcards = _cards2array(infoset.all_handcards['p2'])
    p4_handcards = _cards2array(infoset.all_handcards['p4'])
    teammate_handcards = _cards2array(infoset.all_handcards['p3'])
    
    my_minStepPlayOut = min_steps_play_out(infoset.player_hand_cards, infoset.rank2play)
    my_minStepPlayOut_ = _get_one_hot_array(my_minStepPlayOut, 27)
    
    p2_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p2'], infoset.rank2play)
    p2_minStepPlayOut_ = _get_one_hot_array(p2_minStepPlayOut, 27)
    
    p4_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p4'], infoset.rank2play)
    p4_minStepPlayOut_ = _get_one_hot_array(p4_minStepPlayOut, 27)
    
    teammate_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p3'], infoset.rank2play)
    teammate_minStepPlayOut_ = _get_one_hot_array(teammate_minStepPlayOut, 27)

    other_handcards = _cards2array(infoset.other_hand_cards)

    last_action = _cards2array(infoset.last_move)
    merged_actions = actions_merge(legal_actions, infoset.rank2play)
    num_legal_actions = len(merged_actions)
    if num_legal_actions > MAX_ACTION:
        selected_actions = np.random.choice(merged_actions,MAX_ACTION,replace=False)
        num_legal_actions = MAX_ACTION
    else:
        selected_actions = merged_actions
    my_action_batch = np.zeros((MAX_ACTION, ACTION_SIZE))
    action_ids = np.array([0] * MAX_ACTION)
    for i in range(num_legal_actions):
        action_ids[i],my_action_batch[i, :] = selected_actions[i]

    last_p2_action = _cards2array(
        infoset.last_move_dict['p2'])
    last_p4_action = _cards2array(
        infoset.last_move_dict['p4'])

    p2_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p2'], 27)
    p4_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p4'], 27)

    p2_played_cards = _cards2array(
        infoset.played_cards['p2'])
    p4_played_cards = _cards2array(
        infoset.played_cards['p4'])

    last_teammate_action = _cards2array(
        infoset.last_move_dict['p3'])
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p3'], 27)

    teammate_played_cards = _cards2array(
        infoset.played_cards['p3'])

    rank2play_ = _get_one_hot_rank2play(
        infoset.rank2play)
    
    state_arr = np.hstack((my_handcards,
                           other_handcards,
                           p4_played_cards,
                           p2_played_cards,
                           teammate_played_cards,
                           last_action,
                           last_p4_action,
                           last_p2_action,
                           last_teammate_action,
                           p4_num_cards_left,
                           p2_num_cards_left,
                           teammate_num_cards_left,
                           rank2play_,
                           my_minStepPlayOut_,
                           p4_minStepPlayOut_,
                           p2_minStepPlayOut_,
                           teammate_minStepPlayOut_,
                           p4_handcards,
                           p2_handcards,
                           teammate_handcards,))

    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    
    state_arr = np.hstack([state_arr,z.flatten(),my_action_batch.flatten()])
    
    min_step_all = [my_minStepPlayOut, p2_minStepPlayOut, teammate_minStepPlayOut, p4_minStepPlayOut]

    obs = {
            'position': 'p1',
            'num_legal_actions': num_legal_actions,
            'rank': infoset.rank2play,
            'state': state_arr.astype(np.bool_),
            'action_id': action_ids.astype(np.int16),
            'legal_actions': infoset.legal_actions,
            'min_step_all': np.array(min_step_all, dtype=np.float32)
          }
    
    return obs


def _get_obs_p2(infoset):
    legal_actions = infoset.legal_actions
    my_handcards = _cards2array(infoset.player_hand_cards)

    p1_handcards = _cards2array(infoset.all_handcards['p1'])
    p3_handcards = _cards2array(infoset.all_handcards['p3'])
    teammate_handcards = _cards2array(infoset.all_handcards['p4'])
    
    my_minStepPlayOut = min_steps_play_out(infoset.player_hand_cards, infoset.rank2play)
    my_minStepPlayOut_ = _get_one_hot_array(my_minStepPlayOut, 27)
    
    p1_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p1'], infoset.rank2play)
    p1_minStepPlayOut_ = _get_one_hot_array(p1_minStepPlayOut, 27)
    
    p3_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p3'], infoset.rank2play)
    p3_minStepPlayOut_ = _get_one_hot_array(p3_minStepPlayOut, 27)
    
    teammate_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p4'], infoset.rank2play)
    teammate_minStepPlayOut_ = _get_one_hot_array(teammate_minStepPlayOut, 27)

    other_handcards = _cards2array(infoset.other_hand_cards)

    last_action = _cards2array(infoset.last_move)
    merged_actions = actions_merge(legal_actions, infoset.rank2play)
    num_legal_actions = len(merged_actions)
    if num_legal_actions > MAX_ACTION:
        selected_actions = np.random.choice(merged_actions,MAX_ACTION,replace=False)
        num_legal_actions = MAX_ACTION
    else:
        selected_actions = merged_actions
    my_action_batch = np.zeros((MAX_ACTION, ACTION_SIZE))
    action_ids = np.array([0] * MAX_ACTION)
    for i in range(num_legal_actions):
        action_ids[i],my_action_batch[i, :] = selected_actions[i]

    last_p1_action = _cards2array(
        infoset.last_move_dict['p1'])
    last_p3_action = _cards2array(
        infoset.last_move_dict['p3'])

    p1_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p1'], 27)
    p3_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p3'], 27)

    p1_played_cards = _cards2array(
        infoset.played_cards['p1'])
    p3_played_cards = _cards2array(
        infoset.played_cards['p3'])

    last_teammate_action = _cards2array(
        infoset.last_move_dict['p4'])
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p4'], 27)

    teammate_played_cards = _cards2array(
        infoset.played_cards['p4'])

    rank2play_ = _get_one_hot_rank2play(
        infoset.rank2play)
    
    state_arr = np.hstack((my_handcards,
                           other_handcards,
                           p1_played_cards,
                           p3_played_cards,
                           teammate_played_cards,
                           last_action,
                           last_p1_action,
                           last_p3_action,
                           last_teammate_action,
                           p1_num_cards_left,
                           p3_num_cards_left,
                           teammate_num_cards_left,
                           rank2play_,
                           my_minStepPlayOut_,
                           p1_minStepPlayOut_,
                           p3_minStepPlayOut_,
                           teammate_minStepPlayOut_,
                           p1_handcards,
                           p3_handcards,
                           teammate_handcards,))

    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    
    state_arr = np.hstack([state_arr,z.flatten(),my_action_batch.flatten()])

    min_step_all = [p1_minStepPlayOut, my_minStepPlayOut, p3_minStepPlayOut, teammate_minStepPlayOut]

    obs = {
            'position': 'p2',
            'num_legal_actions': num_legal_actions,
            'rank': infoset.rank2play,
            'state': state_arr.astype(np.bool_),
            'action_id': action_ids.astype(np.int16),
            'legal_actions': infoset.legal_actions,
            'min_step_all': np.array(min_step_all, dtype=np.float32)
          }
    
    return obs


def _get_obs_p3(infoset):
    legal_actions = infoset.legal_actions
    my_handcards = _cards2array(infoset.player_hand_cards)
    
    p2_handcards = _cards2array(infoset.all_handcards['p2'])
    p4_handcards = _cards2array(infoset.all_handcards['p4'])
    teammate_handcards = _cards2array(infoset.all_handcards['p3'])
    
    my_minStepPlayOut = min_steps_play_out(infoset.player_hand_cards, infoset.rank2play)
    my_minStepPlayOut_ = _get_one_hot_array(my_minStepPlayOut, 27)
    
    p2_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p2'], infoset.rank2play)
    p2_minStepPlayOut_ = _get_one_hot_array(p2_minStepPlayOut, 27)
    
    p4_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p4'], infoset.rank2play)
    p4_minStepPlayOut_ = _get_one_hot_array(p4_minStepPlayOut, 27)
    
    teammate_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p3'], infoset.rank2play)
    teammate_minStepPlayOut_ = _get_one_hot_array(teammate_minStepPlayOut, 27)

    other_handcards = _cards2array(infoset.other_hand_cards)

    last_action = _cards2array(infoset.last_move)
    merged_actions = actions_merge(legal_actions, infoset.rank2play)
    num_legal_actions = len(merged_actions)
    if num_legal_actions > MAX_ACTION:
        selected_actions = np.random.choice(merged_actions,MAX_ACTION,replace=False)
        num_legal_actions = MAX_ACTION
    else:
        selected_actions = merged_actions
    my_action_batch = np.zeros((MAX_ACTION, ACTION_SIZE))
    action_ids = np.array([0] * MAX_ACTION)
    for i in range(num_legal_actions):
        action_ids[i],my_action_batch[i, :] = selected_actions[i]

    last_p2_action = _cards2array(
        infoset.last_move_dict['p2'])
    last_p4_action = _cards2array(
        infoset.last_move_dict['p4'])

    p2_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p2'], 27)
    p4_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p4'], 27)

    p2_played_cards = _cards2array(
        infoset.played_cards['p2'])
    p4_played_cards = _cards2array(
        infoset.played_cards['p4'])

    last_teammate_action = _cards2array(
        infoset.last_move_dict['p1'])
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p1'], 27)

    teammate_played_cards = _cards2array(
        infoset.played_cards['p1'])

    rank2play_ = _get_one_hot_rank2play(
        infoset.rank2play)
    
    state_arr = np.hstack((my_handcards,
                           other_handcards,
                           p2_played_cards,
                           p4_played_cards,
                           teammate_played_cards,
                           last_action,
                           last_p2_action,
                           last_p4_action,
                           last_teammate_action,
                           p2_num_cards_left,
                           p4_num_cards_left,
                           teammate_num_cards_left,
                           rank2play_,
                           my_minStepPlayOut_,
                           p2_minStepPlayOut_,
                           p4_minStepPlayOut_,
                           teammate_minStepPlayOut_,
                           p2_handcards,
                           p4_handcards,
                           teammate_handcards,))

    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    
    state_arr = np.hstack([state_arr,z.flatten(),my_action_batch.flatten()])
    
    min_step_all = [teammate_minStepPlayOut, p2_minStepPlayOut, my_minStepPlayOut, p4_minStepPlayOut]

    obs = {
            'position': 'p3',
            'num_legal_actions': num_legal_actions,
            'rank': infoset.rank2play,
            'state': state_arr.astype(np.bool_),
            'action_id': action_ids.astype(np.int16),
            'legal_actions': infoset.legal_actions,
            'min_step_all': np.array(min_step_all, dtype=np.float32)
          }
    
    return obs


def _get_obs_p4(infoset):
    legal_actions = infoset.legal_actions
    my_handcards = _cards2array(infoset.player_hand_cards)
    
    p1_handcards = _cards2array(infoset.all_handcards['p1'])
    p3_handcards = _cards2array(infoset.all_handcards['p3'])
    teammate_handcards = _cards2array(infoset.all_handcards['p4'])
    
    my_minStepPlayOut = min_steps_play_out(infoset.player_hand_cards, infoset.rank2play)
    my_minStepPlayOut_ = _get_one_hot_array(my_minStepPlayOut, 27)
    
    p1_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p1'], infoset.rank2play)
    p1_minStepPlayOut_ = _get_one_hot_array(p1_minStepPlayOut, 27)
    
    p3_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p3'], infoset.rank2play)
    p3_minStepPlayOut_ = _get_one_hot_array(p3_minStepPlayOut, 27)
    
    teammate_minStepPlayOut = min_steps_play_out(infoset.all_handcards['p4'], infoset.rank2play)
    teammate_minStepPlayOut_ = _get_one_hot_array(teammate_minStepPlayOut, 27)

    other_handcards = _cards2array(infoset.other_hand_cards)

    last_action = _cards2array(infoset.last_move)
    merged_actions = actions_merge(legal_actions, infoset.rank2play)
    num_legal_actions = len(merged_actions)
    if num_legal_actions > MAX_ACTION:
        selected_actions = np.random.choice(merged_actions,MAX_ACTION,replace=False)
        num_legal_actions = MAX_ACTION
    else:
        selected_actions = merged_actions
    my_action_batch = np.zeros((MAX_ACTION, ACTION_SIZE))
    action_ids = np.array([0] * MAX_ACTION)
    for i in range(num_legal_actions):
        action_ids[i],my_action_batch[i, :] = selected_actions[i]

    last_p3_action = _cards2array(
        infoset.last_move_dict['p3'])
    last_p1_action = _cards2array(
        infoset.last_move_dict['p1'])

    p3_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p3'], 27)
    p1_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p1'], 27)
    
    p3_played_cards = _cards2array(
        infoset.played_cards['p3'])
    p1_played_cards = _cards2array(
        infoset.played_cards['p1'])

    last_teammate_action = _cards2array(
        infoset.last_move_dict['p2'])
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['p2'], 27)

    teammate_played_cards = _cards2array(
        infoset.played_cards['p2'])

    rank2play_ = _get_one_hot_rank2play(
        infoset.rank2play)
    
    state_arr = np.hstack((my_handcards,
                           other_handcards,
                           p3_played_cards,
                           p1_played_cards,
                           teammate_played_cards,
                           last_action,
                           last_p3_action,
                           last_p1_action,
                           last_teammate_action,
                           p3_num_cards_left,
                           p1_num_cards_left,
                           teammate_num_cards_left,
                           rank2play_,
                           my_minStepPlayOut_,
                           p3_minStepPlayOut_,
                           p1_minStepPlayOut_,
                           teammate_minStepPlayOut_,
                           p3_handcards,
                           p1_handcards,
                           teammate_handcards,))

    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    
    state_arr = np.hstack([state_arr,z.flatten(),my_action_batch.flatten()])
    
    min_step_all = [p1_minStepPlayOut, teammate_minStepPlayOut, p3_minStepPlayOut, my_minStepPlayOut]
    
    obs = {
            'position': 'p4',
            'num_legal_actions': num_legal_actions,
            'rank': infoset.rank2play,
            'state': state_arr.astype(np.bool_),
            'action_id': action_ids.astype(np.int16),
            'legal_actions': infoset.legal_actions,
            'min_step_all': np.array(min_step_all, dtype=np.float32)
          }
    
    return obs

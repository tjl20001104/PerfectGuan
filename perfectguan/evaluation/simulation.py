import multiprocessing as mp
import pickle

from perfectguan.env.game import GameEnv
from perfectguan.env.utils import separator_len
from os import path, remove


def load_card_play_models(card_play_model_path_dict, epoch):
    players = {}

    for position in ['p1', 'p2', 'p3', 'p4']:
        if card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        elif 'rule' in card_play_model_path_dict[position]:
            if card_play_model_path_dict[position] == 'rule_1':
                from .rule_agent import RuleAgent_1
                players[position] = RuleAgent_1()
            elif card_play_model_path_dict[position] == 'rule_2':
                from .rule_agent import RuleAgent_2
                players[position] = RuleAgent_2()
        elif card_play_model_path_dict[position] == 'guanzero':
            from .guanzero_agent import GuanZeroAgent
            players[position] = GuanZeroAgent(position)
        elif card_play_model_path_dict[position] == 'perfectguan':
            from .perfectguan_agent import PerfectGuanAgent
            players[position] = PerfectGuanAgent(position, epoch)
        else:
            raise KeyError('--{} should be chosen in [random, rule_1, \
                           rule_2, guanzero, perfectguan]'.format(position))
    return players


def mp_simulate(card_play_data_list, card_play_model_path_dict, epoch, q):
    players = load_card_play_models(card_play_model_path_dict, epoch)

    env = GameEnv(players)

    for _, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step_h()
        env.reset()

    coop_actual_t1 = 0
    coop_couldve_t1 = 0
    dwarf_actual_t1 = 0
    dwarf_couldve_t1 = 0
    assist_actual_t1 = 0
    assist_couldve_t1 = 0
    
    coop_actual_t2 = 0
    coop_couldve_t2 = 0
    dwarf_actual_t2 = 0
    dwarf_couldve_t2 = 0
    assist_actual_t2 = 0
    assist_couldve_t2 = 0

    coop_actual_t1 = env.coop_counter['p1'][0] + env.coop_counter['p3'][0]
    coop_couldve_t1 = env.coop_counter['p1'][1] + env.coop_counter['p3'][1]
    dwarf_actual_t1 = env.coop_counter['p1'][2] + env.coop_counter['p3'][2]
    dwarf_couldve_t1 = env.coop_counter['p1'][3] + env.coop_counter['p3'][3]
    assist_actual_t1 = env.coop_counter['p1'][4] + env.coop_counter['p3'][4]
    assist_couldve_t1 = env.coop_counter['p1'][5] + env.coop_counter['p3'][5]

    coop_actual_t2 = env.coop_counter['p2'][0] + env.coop_counter['p4'][0]
    coop_couldve_t2 = env.coop_counter['p2'][1] + env.coop_counter['p4'][1]
    dwarf_actual_t2 = env.coop_counter['p2'][2] + env.coop_counter['p4'][2]
    dwarf_couldve_t2 = env.coop_counter['p2'][3] + env.coop_counter['p4'][3]
    assist_actual_t2 = env.coop_counter['p2'][4] + env.coop_counter['p4'][4]
    assist_couldve_t2 = env.coop_counter['p2'][5] + env.coop_counter['p4'][5]
        
    # coop_actual = sum([ac for ac in [v[0] for v in env.coop_counter.values()]])
    # coop_couldve = sum([ac for ac in [v[1] for v in env.coop_counter.values()]])
    # dwarf_actual = sum([ac for ac in [v[2] for v in env.coop_counter.values()]])
    # dwarf_couldve = sum([ac for ac in [v[3] for v in env.coop_counter.values()]])
    # assist_actual = sum([ac for ac in [v[4] for v in env.coop_counter.values()]])
    # assist_couldve = sum([ac for ac in [v[5] for v in env.coop_counter.values()]])
    
    q.put((env.playerAndWins['p1'],
           env.playerAndWins['p2'],
           env.playerAndScore['p1'],
           env.playerAndScore['p2'],
           env.double_downs,
           coop_actual_t1,
           coop_couldve_t1,
           dwarf_actual_t1,
           dwarf_couldve_t1,
           assist_actual_t1,
           assist_couldve_t1,
           coop_actual_t2,
           coop_couldve_t2,
           dwarf_actual_t2,
           dwarf_couldve_t2,
           assist_actual_t2,
           assist_couldve_t2,
         ))


def mp_sim_h(card_play_data_list, card_play_model_path_dict, epoch, role):
    players = load_card_play_models(card_play_model_path_dict, epoch)

    env = GameEnv(players)

    if role in ['p1', 'p2', 'p3', 'p4']:
        print('\n> You are playing as: {} <'.format(role))
    elif role == 'obs4':
        print('\n> You are observing all 4 players')

    for num_games, card_play_data in enumerate(card_play_data_list):
        if role in ['p1', 'p2', 'p3', 'p4', 'obs4']:
            print('\n\u2699\u2699\u2699 Starting Game #{} \u2699\u2699\u2699'.format(
                num_games + 1))
        else:
            print('>>> Initial hands: \np1: {}\np2: {}\np3: {}\np4: {}'.format(
                card_play_data['p1'],
                card_play_data['p2'],
                card_play_data['p3'],
                card_play_data['p4'],
            ))
            print('-' * separator_len)

        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step_h(role, display_cards=True)
        
        if env.game_over:
            print('=' * separator_len)  #\u279C\u21D2\u21F6
            finishingOrderStr = '['
            for fo in env.finishingOrder:
                finishingOrderStr += 'p' + str(fo) + ', '
            finishingOrderStr = finishingOrderStr[:-2]
            finishingOrderStr += ']'
            print('GG! Finishing order: {}\n\
                   : [player: rank] \u21D2 {}\n\
                   : [player: score] \u21D2 {}'.format(
                finishingOrderStr, env.playerAndRank, env.playerAndScore))
            print('=' * separator_len)

        breakpoint()
        env.reset()


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker


def evaluate(p1, p2, p3, p4, epoch, eval_data, num_workers):
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    # Cleaning up misc. stuff, assuming fresh start ALWAYS
    if path.isfile('playerAndRank'):
        remove('playerAndRank')

    if path.isfile('playerAndScore'):
        remove('playerAndScore')

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4}

    num_t1_wins = 0
    num_t2_wins = 0
    num_t1_scores = 0
    num_t2_scores = 0
    num_double_downs = 0
    num_couldve_crushed_t1 = 0
    num_actually_crushed_t1 = 0
    num_couldve_dwarved_t1 = 0
    num_actually_dwarved_t1 = 0
    num_couldve_assisted_t1 = 0
    num_actually_assisted_t1 = 0
    num_couldve_crushed_t2 = 0
    num_actually_crushed_t2 = 0
    num_couldve_dwarved_t2 = 0
    num_actually_dwarved_t2 = 0
    num_couldve_assisted_t2 = 0
    num_actually_assisted_t2 = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []

    for card_play_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_data, card_play_model_path_dict, epoch, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for _ in range(num_workers):
        result = q.get()
        num_t1_wins += result[0]
        num_t2_wins += result[1]
        num_t1_scores += result[2]
        num_t2_scores += result[3]
        num_double_downs += result[4]
        num_actually_crushed_t1 += result[5]
        num_couldve_crushed_t1 += result[6]
        num_actually_dwarved_t1 += result[7]
        num_couldve_dwarved_t1 += result[8]
        num_actually_assisted_t1 += result[9]
        num_couldve_assisted_t1 += result[10]
        num_actually_crushed_t2 += result[11]
        num_couldve_crushed_t2 += result[12]
        num_actually_dwarved_t2 += result[13]
        num_couldve_dwarved_t2 += result[14]
        num_actually_assisted_t2 += result[15]
        num_couldve_assisted_t2 += result[16]

    agent_types = []
    for p in card_play_model_path_dict.keys():
        if card_play_model_path_dict[p] == 'random':
            # agent_types.append('random\U0001F3B2')
            agent_types.append('\U0001F3B2')
        else:
            # agent_types.append('guanzero\U0001F9E0')
            agent_types.append('\U0001F9E0')

    num_games = num_t1_wins + num_t2_wins
    print('\n\u1368 {} Workers simulated {} games \u25B6\u25B6\u25B6\
          \n\u1368 Team A = p1\u27A4{} + p3\u27A4{} \U0001F19A Team B = p2\u27A4{} + p4\u27A4{}\
          \n\u1368 Win Percentage \U0001F449 Team A: {:.1f}% \U0001F19A Team B: {:.1f}%\
          \n\u1368 Score/game \U0001F449 Team A: {:.1f} \U0001F19A Team B: {:.1f}\
          \n\u1368 Double-down Percentage: {:.0f}%\
          \n\u1368 Coop Percentage \U0001F449 Team A: {:.0f}% \U0001F19A Team B: {:.0f}%\
          \n\u1368 Dwarf Percentage \U0001F449 Team A: {:.0f}% \U0001F19A Team B: {:.0f}%\
          \n\u1368 Assist Percentage \U0001F449 Team A: {:.0f}% \U0001F19A Team B: {:.0f}%'.format(
        num_workers, num_games,
        agent_types[0], agent_types[2], agent_types[1], agent_types[3],
        num_t1_wins / num_games * 100, num_t2_wins / num_games * 100,
        num_t1_scores / 14 / num_games, num_t2_scores / 14 / num_games,
        num_double_downs / num_games * 100,
        100 - num_actually_crushed_t1 / num_couldve_crushed_t1 * 100, 100 - num_actually_crushed_t2 / num_couldve_crushed_t2 * 100,
        num_actually_dwarved_t1 / num_couldve_dwarved_t1 * 100, num_actually_dwarved_t2 / num_couldve_dwarved_t2 * 100,
        num_actually_assisted_t1 / num_couldve_assisted_t1 * 100, num_actually_assisted_t2 / num_couldve_assisted_t2 * 100, 
        ))


def eval_h(p1, p2, p3, p4, epoch, eval_data, num_workers, role):  # play_with_human()
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4}

    for card_play_data in card_play_data_list_each_worker:
        mp_sim_h(card_play_data, card_play_model_path_dict, epoch, role)

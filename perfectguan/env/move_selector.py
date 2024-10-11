# return all moves that can beat rivals, moves and rival_move should be same type
import numpy as np
from .utils import cards2rank


def common_handle(moves, rival_move, rival_move_rank=-1, rank2play=-1, ace_as_one=False):
    # Sort rival_move
    rival_move_ranked = cards2rank(rival_move)
    # print('rival_move_ranked: {}'.format(rival_move_ranked))

    if rival_move_rank == -1:
        if len(rival_move_ranked) > 0:
            rival_move_rank = rival_move_ranked[0]
        if rank2play >= 2 and rival_move_rank == rank2play:
            rival_move_rank = 17

    if ace_as_one:
        if rival_move[0] == 2 and rival_move[-1] == 14:
            rival_move_rank = 1

    new_moves = list()
    for move in moves:
        move_ranked = cards2rank(move)
        move_rank = move_ranked[0]
        if rank2play != -1 and move_rank == rank2play:
            move_rank = 17

        if move_rank > rival_move_rank:
            new_moves.append(move)
    return new_moves

def filter_type_1_single(moves, rival_move, rank2play):
    return common_handle(moves, rival_move, rank2play=rank2play)


def filter_type_2_pair(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_3_triple(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_4_bomb(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_45_bomb(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_6_straight_flush(moves, rival_move, rival_move_rank):
    return common_handle(moves, rival_move, rival_move_rank, ace_as_one=True)


def filter_type_46_bomb(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_47_bomb(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_48_bomb(moves, rival_move, rival_move_rank, rank2play):
    return common_handle(moves, rival_move, rival_move_rank=rival_move_rank, rank2play=rank2play)


def filter_type_7_3_2(moves, rival_move_rank, rank2play):  # assumes no wild card
    new_moves = list()
    for move in moves:
        move_ranked = cards2rank(move)
        move_rank = move_ranked[2]
        if move_rank == rank2play:
            move_rank = 17

        if move_rank > rival_move_rank:
            new_moves.append(move)
    return new_moves


def filter_type_8_serial_single(moves, rival_move, rival_move_rank):
    return common_handle(moves, rival_move, rival_move_rank, ace_as_one=True)


def filter_type_9_serial_pair(moves, rival_move, rival_move_rank):
    return common_handle(moves, rival_move, rival_move_rank, ace_as_one=True)


def filter_type_10_serial_triple(moves, rival_move, rival_move_rank):
    return common_handle(moves, rival_move, rival_move_rank, ace_as_one=True)

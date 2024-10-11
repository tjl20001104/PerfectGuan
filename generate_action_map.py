import json

rank2str = {
    1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: 'H', 16: 'B', 17: 'R'}
all_actions = []
base_actions = []

# SINGLE
for i in range(2,18):
    if i == 15:
        continue
    action = [i] * 1
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# DOUBLE
for i in range(2,18):
    if i == 15:
        continue
    action = [i] * 2
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# TRIPLE
for i in range(2,15):
    action = [i] * 3
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# FOUR BOMB
for i in range(2,15):
    action = [i] * 4
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# FIVE BOMB
for i in range(2,15):
    action = [i] * 5
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# SIX BOMB
for i in range(2,15):
    action = [i] * 6
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# SEVEN BOMB
for i in range(2,15):
    action = [i] * 7
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# EIGHT BOMB
for i in range(2,15):
    action = [i] * 8
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# ROCKET
base_actions.append('BBRR')
# STRAIGHT FLUSH
for i in range(1,11):
    action = [i+j for j in range(5)]
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str+'.H')
    base_actions.append(action_str+'.S')
    base_actions.append(action_str+'.D')
    base_actions.append(action_str+'.C')
# TRIPLE WITH PAIR
for i in range(2,15):
    for j in range(2,18):
        if j != i and j != 15:
            action = [i] * 3 + [j] * 2
            action.sort()
            action_list = [rank2str[a] for a in action]
            action_str = ''.join(action_list)
            base_actions.append(action_str)
# SERIAL SINGLE
for i in range(1,11):
    action = [i+j for j in range(5)]
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# SERIAL DOUBLE
for i in range(1,13):
    action = [i+j for j in range(3)] * 2
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# SERIAL TRIPLE
for i in range(1,14):
    action = [i+j for j in range(2)] * 3
    action.sort()
    action_list = [rank2str[a] for a in action]
    action_str = ''.join(action_list)
    base_actions.append(action_str)
# PASS
base_actions.append('PASS')

action_space = {}
for i in range(len(base_actions)):
    action_space[base_actions[i]] = i

with open("action_space.json", "w") as file:
    json.dump(action_space,file,indent=4)

import argparse
import sys
import numpy as np


np.random.seed(8111986)

parser = argparse.ArgumentParser(description='Import and mutate hyperparameters, generating a child population from parents, with mutated and swaped hyperparameters')
parser.add_argument('infile', type=str, help='[str] Input hyperparameter file')
parser.add_argument('outfile', type=str, help='[str] Output hyperparameter file with mutated hyperparameters')
parser.add_argument('--mutation_rate', default=0.5, type=float, help='[float] mutation rate, default = 0.5', required=False)
parser.add_argument('--swap_rate', default=0.5, type=float, help='[float] swap rate, default = 0.5', required=False)
parser.add_argument('--pop_size', default=12, type=int, help='[int] number of children to be generated, default = 12', required=False)
parser.add_argument('--keep_parents', default='no', type=str, help='[yes,no] Whether to keep parents in the child population, default = no', required=False)

args = parser.parse_args()

verbose = 0


hp_names = ['L1', 'L2', 'BETA', 'RHO', 'ACT', 'LR', 'GAMMA', 'OPT', 'LT', 'HS','LB', 'RB', 'KP']

exp_names =  ['L1', 'L2', 'RHO', 'LR']
lin_names =  ['BETA', 'GAMMA', 'KP']

if(args.infile == args.outfile):
    print("Output should not have the same name as infile, exiting.")
    sys.exit()

hp_ranges_file = open('hp_ranges.txt')
hp_ranges = {}

for r_line in hp_ranges_file:
    r_line = r_line.strip().split(' ')
    hp_ranges[r_line[0]] = r_line[1:]

print("hyperparameter ranges",hp_ranges)

infile_for = open(args.infile)
hp_set = []

for hp_line in infile_for:

    hp_values = (hp_line.strip().split(' '))
    hp_dict = dict(zip(hp_names, hp_values))
    hp_dict['KP'] = hp_dict['KP'].split(',')
    hp_set.append(hp_dict)
    

infile_for.close()

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 


def swap_hp(parents):
    children_pop = []
    parent_pair = []
    parent_i = np.random.choice(list(range(len(parents))), size=2)
    for i in parent_i:
        parent_pair.append(parents[i])
    if(verbose>0):
        print("before swap", parent_pair)
    for i in parent_pair[0].keys():
        my_choice = np.random.choice([0,1], size=1, p=[1-args.swap_rate, args.swap_rate])
        if(my_choice == 1):
            hp0 = parent_pair[0][i]
            hp1 = parent_pair[1][i]
            parent_pair[0][i] = hp1
            parent_pair[1][i] = hp0
    if(verbose>0):
        print("after swap", parent_pair)
    return parent_pair

def mutate_pop(parents):
    child_pop = []
    for i in range(args.pop_size):
        my_child = parents[i].copy()
        for hp in my_child.keys():
            my_choice = np.random.choice([0,1], size=1, p=[1-args.mutation_rate, args.mutation_rate])
            if(my_choice == 1):
                my_child[hp] = mutate_hp(my_child[hp],hp)
        child_pop.append(my_child)
        if(verbose>0):
            print("new_mutated_child", my_child)
    return child_pop

                
def mutate_hp(hp_value, hp_name):
    if(hp_name in lin_names):
        if(hp_name != 'KP'):
            new_value = np.random.uniform(float(hp_ranges[hp_name][0]), float(hp_ranges[hp_name][1]))
            new_value = np.round(new_value,8)
        else:
            new_values = []
            for value in hp_value:
                new_value = np.random.uniform(float(hp_ranges[hp_name][0]), float(hp_ranges[hp_name][1]))
                new_value = np.round(new_value,2)
                new_values.append(new_value)
            new_value = new_values
    elif(hp_name in exp_names):
        if(verbose>0):
            print(hp_ranges[hp_name][0], hp_ranges[hp_name][1])
            print(hp_ranges)
        new_value = np.random.choice(powspace(float(hp_ranges[hp_name][0]), float(hp_ranges[hp_name][1]), 10, 20))
        if(verbose>0):
            print("mutated",new_value)
        new_value = np.round(new_value,8)

    else:
        new_value = hp_value
    return new_value

new_pop = []

i=0
while(i<args.pop_size):
    swaped_pair = swap_hp(hp_set.copy())
    if(len(new_pop)<args.pop_size):
        new_pop.append(swaped_pair[0])
        i=i+1



child_pop = mutate_pop(new_pop.copy())

print("before swaping and mutating")
for new_set in hp_set:
    print(list(new_set.values()))


print("after swap")
for new_set in new_pop:
    print(list(new_set.values()))

print("after mutation")
for new_set in child_pop:
    print(list(new_set.values()))


with open(args.outfile, 'w') as outfile:
    for i in child_pop:
        first=True
        for k,v in i.items():
            if(type(v) is list):
                outfile.write(' ')
                wc=0
                for j in v:
                    outfile.write(str(j))
                    wc=wc+1
                    if(wc<len(v)):
                        outfile.write(',')
            else:
                if(first==False):
                    outfile.write(" "+str(v))
                if(first==True):
                    outfile.write(str(v))
                    first=False
        outfile.write('\n')

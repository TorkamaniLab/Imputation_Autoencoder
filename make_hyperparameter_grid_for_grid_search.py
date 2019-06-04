
# coding: utf-8

import itertools

import random #masking

import timeit #measure runtime


def make_new_grid(l1_arr, l2_arr, beta_arr, rho_arr, act_arr, learning_rate_arr, gamma_arr, opt_array, loss_arr, hs_arr, lb_array, rb_array, N=1000):
    print("Building grid search combinations.")

    grid = [l1_arr, l2_arr, beta_arr, rho_arr, act_arr, learning_rate_arr, gamma_arr, opt_array, loss_arr, hs_arr, lb_array, rb_array]
    l = list(itertools.product(*grid))

    print("Extracted", N, "from", len(l), "possible combinations.")
    list_of_random_items = random.sample(l, N)
    idx=0

    file_name = "hyper_parameter_list."+str(N)+".txt"

    print("Saving new grid to", file_name)    
    open(file_name, "w").close()

    while idx < len(list_of_random_items):
        par_list = ''
        for i in range(len(list_of_random_items[idx])):
            par_list += " " + str(list_of_random_items[idx][i])

        with open(file_name, "a") as par_file:
            print(par_list[1:], file=par_file)
        
        idx+=1


#Old version used until may 2019
#act_arr = ['sigmoid','tanh', 'relu']
#l1_arr = [1e-3,1e-4,1e-5,1e-6,1e-1,1e-2,1e-7,1e-8] #RR these are the values for l1 that we want to test in the search grid, should be between zero and 1, near zero
#l2_arr = [1e-3,1e-4,1e-5,1e-6,1e-1,1e-2,1e-7,1e-8]  #RR these are the values for l2 that we want to test in the search grid, should be between zero and 1, near zero
#beta_arr = [0.001, 0.01,0.05,1,2,4,6,8,10] #RR these are the values for beta that we want to test in the search grid, should be greater than zero
#rho_arr = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1.0] #RR these are the values for rho that we want to test in the search grid, should be between 0 and 1
#learning_rate_arr = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#gamma_arr = [0,0.5,1,2,3,5]
#opt_array = ["Adam", "RMSProp", "GradientDescent"]
#loss_arr = ["MSE", "CE", "FL"]

act_arr = ['tanh']
l1_arr = [1e-3,1e-4,1e-5,1e-6,1e-1,1e-2,1e-7,1e-8] #RR these are the values for l1 that we want to test in the search grid, should be between zero and 1, near zero
l2_arr = [0]  #RR these are the values for l2 that we want to test in the search grid, should be between zero and 1, near zero
beta_arr = [1,2,4,6,8,10] #RR these are the values for beta that we want to test in the search grid, should be greater than zero
rho_arr = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1.0] #RR these are the values for rho that we want to test in the search grid, should be between 0 and 1
learning_rate_arr = [0.00001, 0.0001, 0.001]
gamma_arr = [0,0.5,1,2,3,5]
opt_array = ["RMSProp"]
loss_arr = ["FL"]
hs_arr = ['sqrt', '0.10', '0.20', '0.40', '0.60', '0.80', '1']
#new parameters added/edited on june 2019
lb_array = ['0'] #let them as fixed labels to replace them by custom values later (i.e neighbor block sizes)
rb_array = ['0'] #let them as fixed labels to replace them by custom values later (i.e neighbor block sizes)

grid_sizes = [100, 500, 1000, 5000, 10000]

for N in grid_sizes:
    make_new_grid(l1_arr, l2_arr, beta_arr, rho_arr, act_arr,learning_rate_arr, gamma_arr, opt_array, loss_arr, hs_arr, lb_array, rb_array, N)



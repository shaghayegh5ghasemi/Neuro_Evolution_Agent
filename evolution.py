from player import Player
import numpy as np
from config import CONFIG
import copy


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        gaussian_parameter = 1
        noise_w0 = gaussian_parameter*np.random.normal(size=(child.nn.layer_sizes[1], child.nn.layer_sizes[0]))
        child.nn.weights[0] += noise_w0
        noise_w1 = gaussian_parameter*np.random.normal(size=(child.nn.layer_sizes[2], child.nn.layer_sizes[1]))
        child.nn.weights[1] += noise_w1
        noise_b0 = gaussian_parameter*np.random.normal(size=(child.nn.layer_sizes[1], 1))
        child.nn.biases[0] += noise_b0
        noise_b1 = gaussian_parameter*np.random.normal(size=(child.nn.layer_sizes[2], 1))
        child.nn.biases[1] += noise_b1


    def q_tournament_selection(self, players, num_players):
        next_population = []
        q = 2
        for i in range(num_players):
            temp_population = []
            for j in range(q):
                temp_population.append(players[np.random.randint(0, len(players))])
            temp_population.sort(key=lambda x: x.fitness, reverse=True)
            next_population.append(temp_population[0])
        return next_population

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            new_population = []
            p_crossover = 0.9
            p_mutation = 0.2
            for i in range(num_players):
                parents = self.q_tournament_selection(prev_players, 2)
                rand_num = np.random.uniform(0, 1)
                if rand_num <= p_crossover:
                    child1 = Player(self.mode)
                    #child1: weights of first layer (first half from first parent and second half from second parent) and so on
                    child1.nn.weights[0][0:child1.nn.layer_sizes[1]//2] = parents[0].nn.weights[0][0:parents[0].nn.layer_sizes[1]//2]
                    child1.nn.weights[0][child1.nn.layer_sizes[1]//2:] = parents[1].nn.weights[0][parents[1].nn.layer_sizes[1]//2:]
                    #child1: weights of second layer
                    child1.nn.weights[1][0:child1.nn.layer_sizes[2]//2] = parents[0].nn.weights[1][0:parents[0].nn.layer_sizes[2]//2]
                    child1.nn.weights[1][child1.nn.layer_sizes[2]//2:] = parents[1].nn.weights[1][parents[1].nn.layer_sizes[2]//2:]
                    #child1: biases of first layer
                    child1.nn.biases[0][0:child1.nn.layer_sizes[1]//2] = parents[0].nn.biases[0][0:parents[0].nn.layer_sizes[1]//2]
                    child1.nn.biases[0][child1.nn.layer_sizes[1]//2:] = parents[1].nn.biases[0][parents[1].nn.layer_sizes[1]//2:]
                    #child1: biases of second layer
                    child1.nn.biases[1][0:child1.nn.layer_sizes[2]//2] = parents[0].nn.biases[1][0:parents[0].nn.layer_sizes[2]//2]
                    child1.nn.biases[1][child1.nn.layer_sizes[2]//2:] = parents[1].nn.biases[1][parents[1].nn.layer_sizes[2]//2:]

                else:
                    child1 = copy.deepcopy(parents[0])

                if np.random.uniform(0, 1) <= p_mutation:
                    self.mutate(child1)
                new_population.append(child1)

            return new_population

    # use Q_tournament as next population selection
    def next_population_selection(self, players, num_players):
        next_population = []
        q = 5
        for i in range(num_players):
            temp_population = []
            for j in range(q):
                temp_population.append(players[np.random.randint(0, len(players))])
            temp_population.sort(key=lambda x: x.fitness, reverse=True)
            next_population.append(temp_population[0])

        #(additional): plotting [avg, max, min]
        population_info = ""
        sum = 0
        for i in range(len(players)):
            sum += players[i].fitness
        population_info += str(sum/len(players))
        temp = copy.deepcopy(players)
        temp.sort(key=lambda x: x.fitness, reverse=True)
        population_info += " "
        population_info += str(temp[0].fitness)
        population_info += " "
        population_info += str(temp[-1].fitness)
        with open('info.txt', "a") as myfile:
            myfile.write(population_info)
            myfile.write("\n")
        return next_population


import copy

import numpy as np
import random
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.generation = 0
        self.num_params = None

    def uniform_random(self, players, num_players):
        selected_players = []
        for i in range(num_players):
            selected_players.append(players[np.random.randint(0, len(players))])
        return selected_players

    def top_k_selection(self, players, num_players):
        players = sorted(players, key=lambda player: player.fitness, reverse=True)
        return players[: num_players]

    def roulette_wheel_selection(self, players, num_players):
        selected_players = []
        players_range = []
        players = sorted(players, key=lambda player: player.fitness, reverse=True)
        fitness_sum = sum(player.fitness for player in players)

        # make the roulette_wheel
        start_pointer = 0
        for player in players:
            range_width = player.fitness / fitness_sum
            players_range.append((start_pointer, start_pointer + range_width))
            start_pointer += range_width

        # use roulette wheel #num_players time
        for i in range(num_players):
            random_num = np.random.rand()
            for j in range(len(players)):
                if players_range[j][0] <= random_num < players_range[j][1]:
                    selected_players.append(players[j])
                    break

        return selected_players

    def stochastic_universal_sampling_selection(self, players, num_players):
        selected_players = []
        players_range = []
        players = sorted(players, key=lambda player: player.fitness, reverse=True)
        fitness_sum = sum(player.fitness for player in players)

        # make the roulette_wheel
        start_pointer = 0
        for player in players:
            range_width = player.fitness / fitness_sum
            players_range.append((start_pointer, start_pointer + range_width))
            start_pointer += range_width

        # select #num_players with step between them
        step = 1 / num_players
        start_pointer = np.random.rand() * (1 / num_players)
        for i in range(num_players):
            for j in range(len(players)):
                if players_range[j][0] <= start_pointer < players_range[j][1]:
                    selected_players.append(players[j])
                    break
            start_pointer += step

        return selected_players

    def q_tournament_selection(self, players, num_players):
        selected_players = []
        q = 2
        for i in range(num_players):
            temp_population = []
            for j in range(q):
                temp_population.append(players[np.random.randint(0, len(players))])
            temp_population.sort(key=lambda x: x.fitness, reverse=True)
            selected_players.append(temp_population[0])

        return selected_players

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)

        fit_sum = 0
        fit_max = players[0].fitness
        fit_min = players[0].fitness
        for player in players:
            fit_sum += player.fitness
            if player.fitness > fit_max:
                fit_max = player.fitness
            if player.fitness < fit_min:
                fit_min = player.fitness
        fit_avg = fit_sum / len(players)
        line = str(fit_avg) + " " + str(fit_max) + " " + str(fit_min) + "\n"
        with open('learning_curve.txt', "a") as file:
            file.write(line)

        # return self.uniform_random(players, num_players)
        # return self.top_k_selection(players, num_players)
        # return self.roulette_wheel_selection(players, num_players)
        # return self.stochastic_universal_sampling_selection(players, num_players)
        return self.q_tournament_selection(players, num_players)

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        self.generation += 1

        if self.num_params is None:
            p = Player(self.game_mode)
            self.num_params = p.nn.num_params

        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            new_parents = self.uniform_random(prev_players, num_players)
            # new_parents = self.top_k_selection(prev_players, num_players)
            # new_parents = self.roulette_wheel_selection(prev_players, num_players)
            # new_parents = self.stochastic_universal_sampling_selection(prev_players, num_players)
            # new_parents = self.q_tournament_selection(prev_players, num_players)

            crossover_children = self.crossover_control(new_parents, num_players)
            mutatant_children = self.mutation_control(crossover_children)

            return mutatant_children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def crossover_control(self, players, num_players):
        P_CROSSOVER = 0.8

        crossover_children = []
        for i in range(num_players // 2):
            if np.random.rand() < P_CROSSOVER:
                parent1 = players[i * 2]
                parent2 = players[i * 2 + 1]
                child1, child2 = self.crossover(parent1, parent2)
                crossover_children.append(child1)
                crossover_children.append(child2)

        return crossover_children

    def crossover(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent1)

        # weights

        shapes = [layer_weight.shape for layer_weight in parent1.nn.weights]

        parent1_w_genes = np.concatenate([layer_weight.flatten() for layer_weight in parent1.nn.weights])
        parent2_w_genes = np.concatenate([layer_weight.flatten() for layer_weight in parent2.nn.weights])

        # split_index = (flat1.size + 1 // 2)
        split_index = random.randint(0, len(parent1_w_genes) - 1)

        child1_w_genes = np.array(parent1_w_genes[0:split_index].tolist() + parent2_w_genes[split_index:].tolist())
        child2_w_genes = np.array(parent2_w_genes[0:split_index].tolist() + parent1_w_genes[split_index:].tolist())

        child1.nn.weights = self.unflatten(child1_w_genes, shapes)
        child2.nn.weights = self.unflatten(child2_w_genes, shapes)

        # biases

        shapes = [layer_bias.shape for layer_bias in parent1.nn.biases]

        parent1_b_genes = np.concatenate([layer_weight.flatten() for layer_weight in parent1.nn.biases])
        parent2_b_genes = np.concatenate([layer_weight.flatten() for layer_weight in parent2.nn.biases])

        # split_index = (flat1.size + 1 // 2)
        split_index = random.randint(0, len(parent1_b_genes) - 1)

        child1_b_genes = np.array(parent1_b_genes[0:split_index].tolist() + parent2_b_genes[split_index:].tolist())
        child2_b_genes = np.array(parent2_b_genes[0:split_index].tolist() + parent1_b_genes[split_index:].tolist())

        child1.nn.biases = self.unflatten(child1_b_genes, shapes)
        child2.nn.biases = self.unflatten(child2_b_genes, shapes)

        return child1, child2

    def mutation_control(self, children):
        P_MUTATION = 0.5 + (0.5 / self.generation)
        N_BITS_MUTATION = 5 + ((self.num_params - 5) / self.generation)

        mutant_children = []
        for child in children:
            if np.random.rand() < P_MUTATION:
                children.append(self.mutate(child, N_BITS_MUTATION))
            else:
                mutant_children.append(child)

        return mutant_children

    def mutate(self, child, N_BITS_MUTATION):
        shapes_w = [layer_weight.shape for layer_weight in child.nn.weights]
        shapes_b = [layer_weight.shape for layer_weight in child.nn.biases]

        child_w_genes = np.concatenate([layer_weight.flatten() for layer_weight in child.nn.weights])
        child_b_genes = np.concatenate([layer_bias.flatten() for layer_bias in child.nn.biases])

        P_BIT_MUTATION = N_BITS_MUTATION / self.num_params
        for i in range(len(child_w_genes) + len(child_b_genes)):
            if np.random.rand() < P_BIT_MUTATION:
                if i < len(child_w_genes):
                    child_w_genes[i] = np.random.normal()
                else:
                    child_b_genes[i - len(child_w_genes)] = np.random.normal()

        for i in range(len(child_w_genes)):
            if np.random.rand() < P_BIT_MUTATION:
                child_w_genes[i] = np.random.normal()

        child.nn.weights = self.unflatten(child_w_genes, shapes_w)
        child.nn.biases = self.unflatten(child_b_genes, shapes_b)

        return child

    def unflatten(self, flattened, shapes):
        new_array = []
        index = 0
        for shape in shapes:
            size = np.product(shape)
            new_array.append(flattened[index: index + size].reshape(shape))
            index += size
        return new_array


# Test
if __name__ == '__main__':
    pass
    # evolution = Evolution()
    #
    # self.nn = NeuralNetwork(layer_sizes)
    #
    # p1 = Player("Neuroevolution")
    # p2 = Player("Neuroevolution")
    #
    # p1_m = evolution.mutate(p1, 1)
    #
    # print(p1_m.shape)

from __future__ import print_function
from operator import attrgetter
from gaps import image_helpers
from gaps.selection import roulette_selection
from gaps.crossover import Crossover
from gaps.individual import Individual
from gaps.image_analysis import ImageAnalysis
from gaps.plot import Plot
from gaps.progress_bar import print_progress


class GeneticAlgorithm(object):


    TERMINATION_THRESHOLD = 10

    # 构造方法：  以下划线开头的是类的内部方法，一般不会被手动调用
    def __init__(self, image, piece_size, population_size, generations, elite_size=2):
        # 初始化种群： 代数，人口 TODO
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = elite_size
        pieces, rows, columns = image_helpers.flatten_image(image, piece_size, indexed=True)
        # 循环population_size次，每次都将Individual方法调用返回的对象加入到list中
        self._population = [Individual(pieces, rows, columns) for _ in range(population_size)]
        self._pieces = pieces

    # 每个类的方法的第一个参数都是self
    def start_evolution(self, verbose):
        print("=== Pieces:      {}\n".format(len(self._pieces)))

        if verbose:
            plot = Plot(self._image)

        ImageAnalysis.analyze_image(self._pieces)

        fittest = None
        best_fitness_score = float("-inf")
        termination_counter = 0

        for generation in range(self._generations):
            print_progress(generation, self._generations - 1, prefix="=== Solving puzzle: ")

            new_population = []

            # Elitism
            # 取适应度最高的两个图片
            elite = self._get_elite_individuals(elites=self._elite_size)
            new_population.extend(elite)

            # 从种群中随机选择popultation - elite_size个父母
            selected_parents = roulette_selection(self._population, elites=self._elite_size)

            # 通过父母生成子代，加入到new_population中
            for first_parent, second_parent in selected_parents:
                # 交叉互换，生成子代
                crossover = Crossover(first_parent, second_parent)
                crossover.run()
                child = crossover.child()
                new_population.append(child)

            # 从上一代中选出适应度最高的一个
            fittest = self._best_individual()

            # 如果上一代最佳比历史最佳好，则termination_counter += 1,否则替换
            if fittest.fitness <= best_fitness_score:
                termination_counter += 1
            else:
                best_fitness_score = fittest.fitness

            # 如果连续十代都没有更优子代，则退出
            if termination_counter == self.TERMINATION_THRESHOLD:
                print("\n\n=== GA terminated")
                print("=== There was no improvement for {} generations".format(self.TERMINATION_THRESHOLD))
                return fittest

            self._population = new_population

            if verbose:
                plot.show_fittest(fittest.to_image(), "Generation: {} / {}".format(generation + 1, self._generations))

        return fittest

    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        # 适应度在这里被attrgetter调用，会计算适应度并排序
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))

import logging
from typing import Sequence, List, Tuple, Optional
import numpy as np
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.initial_designs import RandomDesign
from ..parameters.protein_base_parameter import ProteinBaseParameter
from emukit.core.optimization.context_manager import ContextManager


_log = logging.getLogger(__name__)


class ProteinBaseGeneticProgrammingOptimizer(AcquisitionOptimizerBase):
	"""
	Optimizes the acquisition function using Genetic programming over a protein sequence (in base representation)
	"""
	def __init__(self, space: ParameterSpace, dynamic:bool = False, num_evolutions: int = 10, 
				 population_size: int = 5, tournament_prob: float = 0.5,
				 p_crossover: float = 0.8, p_mutation: float = 0.05
				) -> None:
		"""
		:param space: The parameter space spanning the search problem (has to consist of a single ProteinParameter).
		:param num_steps: Maximum number of evolutions.
		:param dynamic: allow early stopping to choose number of steps (chooses between 10 and 100 evolutions)
		:param num_init_points: Population size.
		:param tournament_prob: proportion of population randomly chosen from which to choose a tree to evolve
								(larger gives faster convergence but smaller gives better diversity in the population)
		:p_crossover: probability of crossover evolution (if not corssover then just keep the same (reproducton))
		:p_mutation: probability of randomly mutatiaon
		
		"""
		super().__init__(space)
		#check that if parameter space is a single cfg param
		if len(space.parameters)!=1 or not isinstance(space.parameters[0],ProteinBaseParameter):
			raise ValueError("Genetic programming optimizer only for spaces consisting of a single protein parameter")
		self.space = space
		self.p_mutation = p_mutation
		self.p_crossover = p_crossover
		self.num_evolutions = num_evolutions
		self.population_size = population_size
		self.tournament_prob = tournament_prob
		self.dynamic = dynamic
		if self.dynamic:
			self.num_evolutions = 10
		else:
			self.num_evolutions = num_evolutions
		self.length = self.space.parameters[0].length
		self.possible_swaps = self.space.parameters[0].possible_swaps
		
	def _optimize(self, acquisition: Acquisition , context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
		"""
		See AcquisitionOptimizerBase._optimizer for parameter descriptions.
		Optimize an acqusition function using a GA
		"""
		# initialize population of strings
		random_design = RandomDesign(self.space)
		population = random_design.get_samples(self.population_size)
		# clac fitness for current population
		fitness_pop = acquisition.evaluate(population)
		standardized_fitness_pop = fitness_pop / sum(fitness_pop)
		# initialize best location and score so far
		X_max = population[np.argmax(fitness_pop)].reshape(-1,1) 
		acq_max = np.max(fitness_pop).reshape(-1,1) 
		iteration_bests=[]
		_log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
		for step in range(self.num_evolutions):
			_log.info("Performing evolution step {}".format(step))
			# evolve populations
			population = self._evolve(population,standardized_fitness_pop)
			# recalc fitness
			fitness_pop = acquisition.evaluate(population)
			standardized_fitness_pop = fitness_pop / sum(fitness_pop)
			# update best location and score (if found better solution)
			acq_pop_max = np.max(fitness_pop)
			iteration_bests.append(acq_pop_max)
			_log.info("best acqusition score in the new population".format(acq_pop_max))
			if acq_pop_max > acq_max[0][0]:
				acq_max[0][0] = acq_pop_max
				X_max[0] = population[np.argmax(fitness_pop)]
		# if dynamic then keep running (stop when no improvement over 10)
		if self.dynamic:
			stop = False
		else:
			stop = True
		i=10
		while not stop:
			_log.info("Performing evolution step {}".format(step))
			# evolve populations
			population = self._evolve(population,standardized_fitness_pop)
			# recalc fitness
			fitness_pop = acquisition.evaluate(population)
			standardized_fitness_pop = fitness_pop / sum(fitness_pop)
			# update best location and score (if found better solution)
			acq_pop_max = np.max(fitness_pop)
			iteration_bests.append(acq_pop_max)
			_log.info("best acqusition score in the new population".format(acq_pop_max))
			if acq_pop_max > acq_max[0][0]:
				acq_max[0][0] = acq_pop_max
				X_max[0] = population[np.argmax(fitness_pop)]
			if acq_max[0][0]==max(iteration_bests[:-10]):
				stop=True
			# also stop if ran for 100 evolutions in total
			if i==100:
				stop=True
			i+=1
		# return best solution from the whole optimization
		return X_max, acq_max
	
	def _evolve(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
		""" Performs a single evolution of the population, returnign the new population
		:param population: current population of strings (2d-array of strings)
		:param fitness:  fitness values of current points (2d-array)
		:return: a new population of strings (2d-array)
		"""
		# perform genetic operations
		new_pop = np.empty((len(population),1),dtype=object)
		i=0
		while i<len(population): 
			# sample a parent string 
			tournament = self._tournament(population,fitness)
			parent1=population[tournament[0]][0]
			parent2=population[tournament[1]][0]
			# sample to see which operation to use
			if np.random.rand(1)[0] < self.p_crossover:
				# perform crossover
				child1,child2 = self._crossover_then_mutate(parent1,parent2)
				# update two memembers of the population (if enough left)
				new_pop[i][0] = child1
				if i<len(population)-1:
					new_pop[i+1][0] = child2
				i+=2
			else:
				# perform reproduction, use string unmodified
				child1 = self._reproduce_then_mutate(parent1)
				new_pop[i][0]=child1
				i+=1
		return new_pop
	
	def _crossover_then_mutate(self, parent1: str, parent2: str) -> Tuple[str,str]:
		# randomly swap strings up to an index in the two strings
		# first conver to lists
		parent1 = parent1.split(" ")
		parent2 = parent2.split(" ")
		#choose point to start swap from
		crossover_point = np.random.randint(1,self.length)
		#smake swap
		for i in range(0,3*crossover_point):
			temp=parent1[i]
			parent1[i]=parent2[i]
			parent2[i]=temp
		# make mutations
		for i in range(0,self.space.parameters[0].length):
			# see if need to mutate child1 at poistion i
			if np.random.rand(1)[0] < self.p_mutation:
				sample = np.random.choice(self.possible_swaps[i])
				parent1[3*i] = sample[0]
				parent1[3*i+1] = sample[1]
				parent1[3*i+2] = sample[2]
			# see if need to mutate child2 at poistion i
			if np.random.rand(1)[0] < self.p_mutation:
				sample = np.random.choice(self.possible_swaps[i])
				parent2[3*i] = sample[0]
				parent2[3*i+1] = sample[1]
				parent2[3*i+2] = sample[2]
		return " ".join(parent1)," ".join(parent2)      

	def _reproduce_then_mutate(self, parent: str,) -> str:
		# mutate (no crossover)
		parent = parent.split(" ")
		for i in range(0,self.space.parameters[0].length):
			# see if need to mutate child1 at poistion i
			if np.random.rand(1)[0] < self.p_mutation:
				sample = np.random.choice(self.possible_swaps[i])
				parent[3*i] = sample[0]
				parent[3*i+1] = sample[1]
				parent[3*i+2] = sample[2]
		return " ".join(parent)


   
	def _tournament(self, population:np.ndarray, fitness:np.ndarray) -> Tuple[int,int] :
		""" perfom a 'tournament' to select a suitiable parent from pop
		1) sample a sub-population of size tournament_size
		2) return index (in full pop) of the winner (and the second best)
		"""
		# size of tournament
		size = int(self.population_size * self.tournament_prob)
		# sample indicies
		contender_indicies = np.random.randint(self.population_size,size=size)
		contender_fitness = fitness[contender_indicies]
		# get best from this tournament and return their index

		best = contender_indicies[np.argmax(contender_fitness)]
		# set this score to worst and return next best
		contender_fitness[np.argmax(contender_fitness)] = np.min(contender_fitness)
		second_best = contender_indicies[np.argmax(contender_fitness)]
		return best, second_best
	
	  
	 
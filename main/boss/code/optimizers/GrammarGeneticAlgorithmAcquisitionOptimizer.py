import logging
from typing import Sequence, List, Tuple, Optional
import numpy as np
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.initial_designs import RandomDesign
from emukit.core.optimization.context_manager import ContextManager
from ..parameters.cfg_parameter import CFGParameter

_log = logging.getLogger(__name__)

class GrammarGeneticProgrammingOptimizer(AcquisitionOptimizerBase):
	"""
	Optimizes the acquisition function using Genetic programming over a CFG parameters
	"""
	def __init__(self, space: ParameterSpace, dynamic:bool = False, num_evolutions: int = 10, 
				 population_size: int = 5, tournament_prob: float = 0.5,
				 p_crossover: float = 0.8, p_mutation: float = 0.05
				) -> None:
		"""
		:param space: The parameter space spanning the search problem (has to consist of a single CFGParameter).
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
		if len(space.parameters)!=1 or not isinstance(space.parameters[0],CFGParameter):
			raise ValueError("Genetic programming optimizer only for spaces consisting of a single cfg parameter")
		self.grammar = space.parameters[0].grammar
		self.max_length = space.parameters[0].max_length
		self.min_length =space.parameters[0].min_length
		self.space = space
		self.p_mutation = p_mutation
		self.p_crossover = p_crossover
		self.dynamic = dynamic
		if self.dynamic:
			self.num_evolutions = 10
		else:
			self.num_evolutions = num_evolutions
		self.population_size = population_size
		self.tournament_prob = tournament_prob

		
		
	def _optimize(self, acquisition: Acquisition , context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
		"""
		See AcquisitionOptimizerBase._optimizer for parameter descriptions.

		Optimize an acqusition function using a GA
		"""
		# initialize population of tree

		random_design = RandomDesign(self.space)
		population = random_design.get_samples(self.population_size)
		# clac fitness for current population
		fitness_pop = acquisition.evaluate(unparse(population))
		standardized_fitness_pop = fitness_pop / sum(fitness_pop)
		# initialize best location and score so far
		X_max = np.zeros((1,1),dtype=object)
		X_max[0] = unparse(population[np.argmax(fitness_pop)])
		acq_max = np.max(fitness_pop).reshape(-1,1) 
		iteration_bests=[]
		_log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
		for step in range(self.num_evolutions):
			_log.info("Performing evolution step {}".format(step))
			# evolve populations
			population = self._evolve(population,standardized_fitness_pop)
			# recalc fitness
			fitness_pop = acquisition.evaluate(unparse(population))
			standardized_fitness_pop = fitness_pop / sum(fitness_pop)
			# update best location and score (if found better solution)
			acq_pop_max = np.max(fitness_pop)
			iteration_bests.append(acq_pop_max)
			_log.info("best acqusition score in the new population".format(acq_pop_max))
			if acq_pop_max > acq_max[0][0]:
				acq_max[0][0] = acq_pop_max
				X_max[0] = unparse(population[np.argmax(fitness_pop)])
		# if dynamic then keep running (stop when no improvement over most recent 10 populations)
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
			fitness_pop = acquisition.evaluate(unparse(population))
			standardized_fitness_pop = fitness_pop / sum(fitness_pop)
			# update best location and score (if found better solution)
			acq_pop_max = np.max(fitness_pop)
			iteration_bests.append(acq_pop_max)
			_log.info("best acqusition score in the new population".format(acq_pop_max))
			if acq_pop_max > acq_max[0][0]:
				acq_max[0][0] = acq_pop_max
				X_max[0] = unparse(population[np.argmax(fitness_pop)])
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
		:param population: current population of trees (2d-array of strings)
		:param fitness:  fitness values of current points (2d-array)
		:return: a new population of trees (2d-array)
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
				# perform crossover (if there exists a subtree with a shared head)
				# if does not exisit then repeat
				# also check not already got these trees in the population
				child1,child2 = self._crossover(parent1,parent2)
				# if unable to find shared subtree
				if (child1 is False):
					pass
				else:
					# update two memembers of the population (if enough left)
					# and if tree not too long or already in this pop
					if (self.check_length(child1)):
						new_pop[i][0] = child1
						i+=1
						if (i<len(population))  and (self.check_length(child2)):
							new_pop[i][0] = child2
							i+=1
			else:
				# perform reproduction and mutation
				child1 = self._reproduce_then_mutate(parent1)
				if (not self.check_length(child1)):
					pass
				else:
					new_pop[i][0]=child1
					i+=1
		return new_pop
	


	def _reproduce_then_mutate(self, parent: str,) -> str:
		# mutate (no crossover)
		# randomly choose a subtree from the parent and replace
		# with a new randomly generated subtree
		#
		# choose subtree to delete
		subtree_node, subtree_index = rand_subtree(parent,self.grammar)
		# chop out subtree
		pre,subtree,post = remove_subtree(parent,subtree_index)
		# generate new trees until generate one with a subtree starting with the head of the desired subtree swap
		found = False
		while not found:
			# generate new tree
			donor = self.space.parameters[0].sample_uniform(1)[0][0]
			# see if have subtree starting with the head of the replacement subtree
			new_subtree_index = rand_subtree_fixed_head(donor,subtree_node)
			if new_subtree_index is not False:
				found=True
		_,new_subtree,_ = remove_subtree(donor,new_subtree_index)
		# return mutated tree
		return pre + new_subtree + post






	def _crossover(self, parent1: str, parent2: str) -> Tuple[str,str]:
		# randomly swap subtrees in two trees
		# if no suitiable subtree exists then return False
		subtree_node, subtree_index = rand_subtree(parent1,self.grammar)
		# chop out subtree
		pre, sub, post = remove_subtree(parent1,subtree_index)
		# sample subtree from donor
		donor_subtree_index = rand_subtree_fixed_head(parent2,subtree_node)
		# if no subtrees with right head node return False
		if not donor_subtree_index:
			return False,False
		else:
			donor_pre, donor_sub, donor_post = remove_subtree(parent2,donor_subtree_index)
			# return the two new tree
			child_1 = pre + donor_sub + post 
			child_2 = donor_pre + sub + donor_post
			return child_1, child_2

   
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

	# Helper function to count length of tree (# terminals)
	def check_length(self,tree):
		length=0
		for t in self.grammar.terminals:
			length+=tree.count(t + ")")
		if (length<=self.max_length) and (length>=self.min_length):
			return True
		else:
			return False

# helper function to swap between parse trees and strings
# e.g '2 + 1' <--- '(S (S (T 2)) (ADD +) (T 1))'	  
def unparse(tree):
	string=[]
	temp=""
	# perform single pass of tree
	for char in tree:
		if char==" ":
			temp=""
		elif char==")":
			if temp[-1]!= ")":
				string.append(temp)
			temp+=char
		else:
			temp+=char
	return " ".join(string)
unparse = np.vectorize(unparse) 




#helper function to choose a random subtree in a given tree
# returning the parent node of the subtree and its index
def rand_subtree(tree,grammar) -> int:
	# single pass through tree (stored as string) to look for the location of swappable_non_terminmals
	split_tree = tree.split(" ")
	swappable_indicies=[i for i in range(0,len(split_tree)) if split_tree[i][1:] in grammar.swappable_nonterminals]
	# randomly choose one of these non-terminals to replace its subtree
	r = np.random.randint(1,len(swappable_indicies))
	chosen_non_terminal = split_tree[swappable_indicies[r]][1:]
	chosen_non_terminal_index = swappable_indicies[r]
	# return chosen node and its index
	return chosen_non_terminal, chosen_non_terminal_index

# helper function to choose a random subtree from a given tree with a specific head node
# if no such subtree then return False, otherwise return the index of the subtree
def rand_subtree_fixed_head(tree, head_node) -> int:
	# single pass through tree (stored as string) to look for the location of swappable_non_terminmals
	split_tree = tree.split(" ")
	swappable_indicies=[i for i in range(0,len(split_tree)) if split_tree[i][1:]==head_node]
	if len(swappable_indicies)==0:
		# no such subtree
		return False
	else:
		# randomly choose one of these non-terminals 
		r = np.random.randint(1,len(swappable_indicies)) if len(swappable_indicies)>1 else 0
		chosen_non_terminal_index = swappable_indicies[r]
		return chosen_non_terminal_index

# helper function to remove a subtree from a tree (given its index)
# returning the str before and after the subtree
# i.e '(S (S (T 2)) (ADD +) (T 1))'
# becomes '(S (S (T 2)) ', '(T 1))'  after removing (ADD +)
def remove_subtree(tree,index)  -> Tuple[str,str,str]:
	split_tree = tree.split(" ")
	pre_subtree = " ".join(split_tree[:index])+" "
	#  get chars to the right of split
	right = " ".join(split_tree[index+1:])
	# remove chosen subtree
	# single pass to find the bracket matching the start of the split
	counter,current_index=1,0
	for char in right:
		if char=="(":
			counter+=1
		elif char==")":
			counter-=1
		if counter==0:
			break
		current_index+=1
	# retrun string after remover tree
	post_subtree = right[current_index+1:]
	# get removed tree
	removed = "".join(split_tree[index]) +" "+right[:current_index+1]
	return (pre_subtree, removed, post_subtree)

	 
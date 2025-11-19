from typing import List, Any, Dict, Callable
import numpy as np
import random

Progenitor = Dict[str, Any]
Genome = List[int]
Population = List[Genome]

# Convert the array of 1s and 0s into a DataFrame of coordinates
def prepare_progenitor(arr: np.ndarray, footprint_size: int = 5) -> Progenitor:
    if footprint_size % 2 == 0:
        raise ValueError("footprint_size must be odd")
    
    indices = np.argwhere(arr == 1)
    footprint = np.ones((footprint_size, footprint_size), dtype=int)
    df_d = {
        'arr': arr,
        'footprint': footprint,
        'row': indices[:, 0],
        'col': indices[:, 1]
    }
    return df_d

# Generate an initial genome by randomly selecting n unique indices from the progenitor DataFrame
# Each index corresponds to a row in the progenitor DataFrame
def generate_genome_random(progenitor: Progenitor) -> Genome:
    rng = np.random.default_rng()
    n = np.random.randint(1, len(progenitor['row'])//progenitor['footprint'].shape[0]**2)
    return list(rng.choice(len(progenitor['row']), size=n, replace=False))

def is_placement_valid(new_gene: int, genome: Genome, progenitor: Progenitor) -> bool:
    footprint = progenitor['footprint']
    hw = (footprint.shape[0]-1) // 2
    
    new_r = progenitor['row'][new_gene]
    new_c = progenitor['col'][new_gene]
    
    for gene in genome:
        r = progenitor['row'][gene]
        c = progenitor['col'][gene]
        
        if abs(new_r - r) <= hw and abs(new_c - c) <= hw:
            return False
    return True

# Generate a genome by selecting n unique indices from the progenitor DataFrame
# Ensures that no two selected points are within the footprint distance of each other by using a greedy algorithm
def generate_genome_constrained(progenitor: Progenitor) -> Genome:
    points = list(range(len(progenitor['row'])))
    random.shuffle(points)
    
    genome = []
    for point in points:
        if is_placement_valid(point, genome, progenitor):
            genome.append(point)
    
    return genome
        

def fitness(genome: Genome, progenitor: Progenitor) -> float:
    arr = progenitor['arr']
    
    footprint = progenitor['footprint']
    hw = (footprint.shape[0]-1) // 2
    
    # From genome, get a matrix which counts the number of times a point is occupied
    occupation_kernel = np.zeros_like(arr)
    
    for gene in genome:
        r = progenitor['row'][gene]
        c = progenitor['col'][gene]
        
        rl = r - hw
        rh = r + hw
        cl = c - hw
        ch = c + hw
        
        # Clamp the rows and columns to be within the bounds of arr
        for i in [rl, rh]:
            if i < 0:
                i = 0
            elif i >= arr.shape[0]:
                i = arr.shape[0] - 1
        
        for j in [cl, ch]:
            if j < 0:
                j = 0
            elif j >= arr.shape[1]:
                j = arr.shape[1] - 1
        
        occupation_kernel[rl:rh+1, cl:ch+1] += 1
        
    occupation_kernel = occupation_kernel * arr
    
    # Subtract 1 from all occupied cells to account for single occupancy being neutral
    # Any cell with more than 1 occupancy will have a negative fitness contribution
    # Cells with 0 occupancy remain 0
    occupation_kernel_score = np.where(occupation_kernel > 0, occupation_kernel-1, 0)
    f = np.sum(occupation_kernel_score * -1)
    
    # Count number of 1s in arr that are within a footprint of any occupied cell
    # Each such cell contributes positively to fitness
    coverage = np.where(occupation_kernel > 0, 1, 0)
    f += np.sum(coverage)
    
    if f < 0:
        f = 1
    return f
    
def generate_population(progenitor: Progenitor, pop_size: int) -> Population:
    return [generate_genome_random(progenitor) for _ in range(pop_size)]

def select_parents_random(population: Population, progenitor: Progenitor, num_parents: int, fitness_func: Callable) -> Population:
    return random.choices(
        population=population,
        weights=[fitness_func(genome, progenitor) for genome in population],
        k=num_parents
    )
    
def select_parents(population: Population, progenitor: Progenitor, num_parents: int, fitness_func: Callable) -> Population:
    sorted_population = sorted(population, key=lambda g: fitness_func(g, progenitor), reverse=True)
    return sorted_population[:num_parents]
    
def mutate_genome(genome: Genome, mutation_rate: float, progenitor: Progenitor) -> Genome:
    new_genome = genome.copy()
    
    n = len(new_genome)
    if random.random() < mutation_rate:
        # Add a gene
        new_gene = random.choice(list(range(len(progenitor['row']))))
        while new_gene in new_genome:
            new_gene = random.choice(list(range(len(progenitor['row']))))
        new_genome.append(new_gene)
    
    for i in range(len(new_genome)):
        if random.random() < mutation_rate:            
            new_gene = random.choice(list(range(len(progenitor['row']))))
            
            while new_gene in new_genome:
                new_gene = random.choice(list(range(len(progenitor['row']))))
                
            new_genome[i] = new_gene
    return new_genome

def mutate_genome_constrained(genome: Genome, mutation_rate: float, progenitor: Progenitor) -> Genome:
    new_genome = genome.copy()
    
    if random.random() < mutation_rate:
        available_genes = [g for g in range(len(progenitor['row'])) if g not in new_genome]
        

#Generate new population only mutated from parents / no crossover
def populate(parents: Population, progenitor: Progenitor, pop_size: int, mutation_rate: float) -> Population:
    new_population = parents.copy()
    while len(new_population) < pop_size:
        parent = random.choice(parents)
        child = mutate_genome(parent, mutation_rate, progenitor)
        new_population.append(child)
    return new_population

def populate_with_crossover(
    parents: Population, 
    progenitor: Progenitor, 
    pop_size: int, 
    mutation_rate: float,
    fitness_func: Callable,
    crossover_rate: float = 0.7,
) -> Population:
    """Generate population using both crossover and mutation"""
    new_population = parents.copy()
    
    while len(new_population) < pop_size:
        if random.random() < crossover_rate and len(parents) >= 2:
            # Crossover: select two parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Get their fitness for weighted crossover
            fitness1 = fitness_func(parent1, progenitor)
            fitness2 = fitness_func(parent2, progenitor)
            
            child = crossover_weighted(parent1, parent2, fitness1, fitness2)
        else:
            # Mutation only: select one parent
            parent = random.choice(parents)
            child = parent.copy()
        
        # Always apply potential mutation
        child = mutate_genome(child, mutation_rate, progenitor)
        new_population.append(child)
    
    return new_population

def run_evolution(
    progenitor: Progenitor,
    pop_size: int,
    num_generations: int,
    num_parents: int,
    mutation_rate: float,
    max_fitness: float,
    fitness_func: Callable,
):
    
    population = generate_population(progenitor, pop_size)
    
    best_fitness = 0.0
    all_time_best_fitness = 0.0
    generation = 0
    
    while best_fitness < max_fitness and generation < num_generations:
        parents = select_parents(population, progenitor, num_parents, fitness_func)
        population = populate_with_crossover(parents, progenitor, pop_size, mutation_rate, fitness_func)
        
        best_genome = max(population, key=lambda g: fitness_func(g, progenitor))
        best_fitness = fitness_func(best_genome, progenitor)
        
        if best_fitness > all_time_best_fitness:
            all_time_best_fitness = best_fitness
            all_time_best_genome = best_genome
            
        print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
        generation += 1
        
    best_genome = max(population, key=lambda g: fitness_func(g, progenitor))
    return best_genome, all_time_best_genome


def genome2coords(genome: Genome, progenitor: Progenitor) -> List[tuple]:
    return [(progenitor['row'][gene], progenitor['col'][gene]) for gene in genome]

def crossover_two_parent(parent1: Genome, parent2: Genome) -> Genome:
    """Combine genes from two parents"""
    # Take random subset from each parent
    all_genes = set(parent1 + parent2)
    
    # Remove duplicates and create child
    child_size = random.randint(
        min(len(parent1), len(parent2)), 
        max(len(parent1), len(parent2))
    )
    
    child = random.sample(list(all_genes), min(child_size, len(all_genes)))
    return child

def crossover_weighted(parent1: Genome, parent2: Genome, fitness1: float, fitness2: float) -> Genome:
    """Weighted crossover - better parent contributes more genes"""
    total_fitness = fitness1 + fitness2
    p1_weight = fitness1 / total_fitness if total_fitness > 0 else 0.5
    
    all_genes = set(parent1 + parent2)
    child = []
    
    # Add genes from parent1 with probability proportional to its fitness
    for gene in parent1:
        if random.random() < p1_weight:
            child.append(gene)
    
    # Add genes from parent2 with remaining probability
    for gene in parent2:
        if gene not in child and random.random() < (1 - p1_weight):
            child.append(gene)
    
    return child
    
if __name__ == "__main__":
    import pickle
    
    with open('./arr.pkl', 'rb') as f:
        arr = pickle.load(f)
        
    df_arr = prepare_progenitor(arr)
    
    _, best_genome = run_evolution(
        progenitor=df_arr,
        pop_size=10000,
        num_generations=400,
        num_parents=20,
        mutation_rate=0.1,
        max_fitness=np.sum(arr)*0.9,
        fitness_func=fitness
    )
    
    coords = genome2coords(best_genome, df_arr)
    
    with open('coords_genome.pkl', 'wb') as f:
        pickle.dump(coords, f)
    
    print(coords)



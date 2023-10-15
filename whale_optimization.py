import numpy as np
import random

class WhaleOptimization():

    def __init__(self, opt_func, constraints, nsols, b, a, a_step, maximize=False):
        self._opt_func = opt_func
        self._constraints = constraints
        self._sols = self._init_solutions(nsols) 
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []
        self._T = self._a / self._a_step
        
    def get_solutions(self):
        """return solutions"""
        return self._sols
                                                                  
    def optimize(self, t):
        """solutions randomly encircle, search or attack"""
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0] 
        #include best solution in next generation solutions
        new_sols = [best_sol]
                                                                 
        for s in ranked_sol[1:]:
            if np.random.uniform(0.0, 1.0) > 0.5:                                      
                A = self._compute_A()                                                     
                norm_A = np.linalg.norm(A)                                                
                if norm_A < 1.0:                                                          
                    new_s = self._encircle(s, best_sol, A)                                
                else:                                                                     
                    ###select random sol                                                  
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]       
                    new_s = self._search(s, random_sol, A)                                
            else:   
                new_s = self._attack(s, best_sol)                                         
            new_sols.append(self._constrain_solution(new_s))

        self._sols = np.stack(new_sols)
        # self._a -= self._a_step
        # 余弦收敛因子
        r = np.random.uniform(0.0, 1.0)
        a_t = 2 * np.cos((np.pi/2)* ((t+1) / self._T))** 0.5
        a_r = 2 * np.cos((np.pi * r) / 2)** 0.5
        if a_t <= a_r:
            self._a = a_t
        else:
            self._a = a_r

    # 混沌映射序列函数
    def _logistic_map(self, num_iterations, lb, ub):
        x = np.zeros(num_iterations)
        tmp = random.random() 
        x[0] = lb + (ub - lb) * tmp
        for i in range(1, num_iterations):
            tmp = 4 * tmp * (1 - tmp)
            x[i] = lb + (ub - lb) * tmp
        return x
    
    def tent_map(self, x, r):
        if x < 0.5:
            return r * x
        else:
            return r * (1 - x)
    
    def initialize_population(self, num_populations, num_variables, variable_ranges, r):
        populations = []
        for _ in range(num_populations):
            population = []
            for i in range(num_variables):
                var_min, var_max = variable_ranges[i]
                random_value = random.uniform(0, 1)
                x = self.tent_map(random_value, r)
                variable = var_min + (var_max - var_min) * x
                population.append(variable)
            populations.append(population)
        return populations

    def _init_solutions(self, nsols):
        """initialize solutions uniform randomly in space"""
        sols = []
        # for c in self._constraints:
        #     sols.append(np.random.uniform(c[0], c[1], size=nsols))
        
        # Logistic混沌映射初始化
        # for c in self._constraints:
        #     # 生成混沌映射序列
        #     chaotic_sequence = self._logistic_map(nsols, c[0], c[1]) 
        #     sols.append(chaotic_sequence)
        # print('sols = ', sols)                                                                    
        # sols = np.stack(sols, axis=-1)
        # print('sols11 = ', type(sols))  
        num_variables = 7  # 变量数量
        r = 2 # 控制参数              
        populations = self.initialize_population(nsols, num_variables, self._constraints, r)
        
        sols = np.array(populations)
        
        return sols

    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]    
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        """find best solution"""
        fitness = self._opt_func(self._sols[:, 0], self._sols[:, 1], self._sols[:, 2],self._sols[:, 3], self._sols[:, 4], self._sols[:, 5], self._sols[:, 6])
        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]
   
        #best solution is at the front of the list
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])

        return [ s[1] for s in ranked_sol] 

    def print_best_solutions(self):
        b_sol = []
        print('generation best solution history')
        print('([fitness], [solution])')
        for s in self._best_solutions:
            b_sol.append(s[0])
            print(s)
        print('\n')
        print('best solution')
        print('([fitness], [solution])')
        print(sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0])
        
        return b_sol

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=7)
        return (2.0*np.multiply(self._a, r))-self._a

    def _compute_C(self):
        return 2.0*np.random.uniform(0.0, 1.0, size=7)
                                                                 
    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol*(0.4+0.5*self._a) - np.multiply(A, D)
                                                                 
    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        D = np.linalg.norm(np.multiply(C, best_sol)  - sol)
        return D

    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol*(0.4+0.5*self._a) - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol)    

    def _attack(self, sol, best_sol):
        D = np.linalg.norm(best_sol - sol)
        L = np.random.uniform(-1.0, 1.0, size=7)
        return np.multiply(np.multiply(D,np.exp(self._b*L)), np.cos(2.0*np.pi*L)) + best_sol*(0.4+0.5*self._a)

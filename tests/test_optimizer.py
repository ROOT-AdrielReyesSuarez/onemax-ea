import unittest
import numpy as np
from src.environment import DynamicOneMaxEnv
from src.optimizer import OnePlusOneEA

class TestOnePlusOneEA(unittest.TestCase):
    def test_mutation_rate_and_flips(self):
        n = 100
        env = DynamicOneMaxEnv(length=n)
        optimizer = OnePlusOneEA(env=env)
        
        parent = np.zeros(n, dtype=np.int8)
        
        # Realizar un gran número de mutaciones para evaluar la media estadísticamente
        num_mutations = 2000
        total_flips = 0
        
        for _ in range(num_mutations):
            child = optimizer.mutate(parent)
            # Como el padre es todo ceros, la suma del hijo es el número de bits invertidos (0 -> 1)
            flips = np.sum(child)
            total_flips += flips
            
        mean_flips = total_flips / num_mutations
        
        # La media teórica es n * p = 100 * 0.01 = 1.0
        # Establecemos un rango de confianza estadística holgado para evitar falsos negativos en los tests
        self.assertAlmostEqual(mean_flips, 1.0, delta=0.15, 
                               msg=f"La tasa de mutación promedio ({mean_flips}) diverge de la teórica (1.0).")

    def test_fitness_monotonicity(self):
        n = 100
        env = DynamicOneMaxEnv(length=n)
        optimizer = OnePlusOneEA(env=env)
        
        _, history = optimizer.run_static(max_iterations=500)
        
        # Verificar que el fitness de la población nunca decrece
        for i in range(1, len(history)):
            self.assertGreaterEqual(history[i], history[i-1], 
                                    msg=f"El fitness decreció de {history[i-1]} a {history[i]} en el paso {i}.")

    def test_static_convergence(self):
        n = 100
        # Realizar 5 ejecuciones independientes para asegurar robustez estadística frente a la aleatoriedad
        runs = 5
        successes = 0
        
        for _ in range(runs):
            env = DynamicOneMaxEnv(length=n)
            optimizer = OnePlusOneEA(env=env)
            _, history = optimizer.run_static(max_iterations=1000)
            
            # Comprobar si converge cerca o llega al óptimo (ej. >= 98)
            if history[-1] >= 98:
                successes += 1
                
        # Al menos el 80% de las ejecuciones (4 de 5) deben converger eficientemente en 1000 generaciones
        self.assertGreaterEqual(successes, 4, 
                                 msg=f"Convergencia deficiente: solo {successes} de {runs} ejecuciones alcanzaron un fitness >= 98.")

if __name__ == '__main__':
    unittest.main()

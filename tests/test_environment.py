import unittest
import numpy as np
from src.environment import DynamicOneMaxEnv

class TestDynamicOneMaxEnv(unittest.TestCase):
    def test_initialization(self):
        # Longitud por defecto (100)
        env = DynamicOneMaxEnv()
        self.assertEqual(env.length, 100)
        self.assertEqual(env.optimal_vector.shape, (100,))
        self.assertTrue(np.all((env.optimal_vector == 0) | (env.optimal_vector == 1)))
        
        # Longitud personalizada
        env_custom = DynamicOneMaxEnv(length=50)
        self.assertEqual(env_custom.length, 50)
        self.assertEqual(env_custom.optimal_vector.shape, (50,))
        
        # Casos de error
        with self.assertRaises(ValueError):
            DynamicOneMaxEnv(length=0)
        with self.assertRaises(ValueError):
            DynamicOneMaxEnv(length=-10)

    def test_get_fitness(self):
        length = 10
        env = DynamicOneMaxEnv(length=length)
        
        # Solución perfecta: idéntica al óptimo
        perfect_sol = env.optimal_vector.copy()
        self.assertEqual(env.get_fitness(perfect_sol), length)
        
        # Solución completamente opuesta
        opposite_sol = 1 - perfect_sol
        self.assertEqual(env.get_fitness(opposite_sol), 0)
        
        # Solución con tipo de dato diferente (por ejemplo, lista estándar en vez de numpy array)
        list_sol = list(perfect_sol)
        self.assertEqual(env.get_fitness(list_sol), length)
        
        # Solución de dimensiones incorrectas
        bad_sol = np.zeros(length + 1, dtype=np.int8)
        with self.assertRaises(ValueError):
            env.get_fitness(bad_sol)

    def test_change_environment(self):
        length = 100
        env = DynamicOneMaxEnv(length=length)
        
        # Guardar copia del óptimo antes de cambiar
        original_optimal = env.optimal_vector.copy()
        
        # Perturbar con k = 20
        k = 20
        indices_flipped = env.change_environment(k)
        
        # Verificar que se cambiaron exactamente k elementos
        self.assertEqual(len(indices_flipped), k)
        self.assertEqual(len(np.unique(indices_flipped)), k)
        
        # Verificar que los índices están en rango [0, length - 1]
        self.assertTrue(np.all((indices_flipped >= 0) & (indices_flipped < length)))
        
        # Verificar que los bits cambiaron exactamente en esos índices
        for idx in range(length):
            if idx in indices_flipped:
                self.assertNotEqual(env.optimal_vector[idx], original_optimal[idx])
            else:
                self.assertEqual(env.optimal_vector[idx], original_optimal[idx])
                
        # Verificar que el fitness de la solución original frente al nuevo óptimo es exactamente length - k
        self.assertEqual(env.get_fitness(original_optimal), length - k)
        
        # Perturbación con k = 0 (no debería hacer nada)
        indices_flipped_zero = env.change_environment(0)
        self.assertEqual(len(indices_flipped_zero), 0)
        self.assertEqual(env.get_fitness(original_optimal), length - k)

        # Casos límite y de error
        with self.assertRaises(ValueError):
            env.change_environment(-1)
        with self.assertRaises(ValueError):
            env.change_environment(length + 1)

if __name__ == '__main__':
    unittest.main()

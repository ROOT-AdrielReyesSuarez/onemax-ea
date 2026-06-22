import unittest
import numpy as np
from src.environment import DynamicOneMaxEnv
from src.optimizer import OnePlusOneEA
from src.simulation import ROOTSimulation

class TestROOTSimulation(unittest.TestCase):
    def test_initialization(self):
        env = DynamicOneMaxEnv(length=100)
        optimizer = OnePlusOneEA(env=env)
        sim = ROOTSimulation(
            env=env,
            optimizer=optimizer,
            acceptability_threshold=80,
            freeze_threshold=95,
            change_frequency=500,
            k_max=15
        )
        self.assertEqual(sim.mu, 80)
        self.assertEqual(sim.freeze_threshold, 95)
        self.assertEqual(sim.change_frequency, 500)
        self.assertEqual(sim.k_max, 15)
        self.assertEqual(len(sim.survival_times), 0)
        self.assertEqual(len(sim.fitness_history), 0)

    def test_run_generates_metrics(self):
        env = DynamicOneMaxEnv(length=20)
        optimizer = OnePlusOneEA(env=env)
        sim = ROOTSimulation(
            env=env,
            optimizer=optimizer,
            acceptability_threshold=15,
            freeze_threshold=18,
            change_frequency=50,  # Frecuencia más corta para acelerar la prueba
            k_max=2
        )
        
        target_changes = 5
        sts = sim.run(target_env_changes=target_changes, verbose=False)
        
        # Comprobar que se registraron los cambios especificados
        self.assertEqual(sim.env_changes, target_changes)
        
        # El historial de fitness debe ser del tamaño correcto (cambios * frecuencia + 1)
        self.assertEqual(len(sim.fitness_history), target_changes * 50 + 1)
        
        # Todos los tiempos de supervivencia acumulados en la simulación deben ser no negativos
        for st in sts:
            self.assertGreaterEqual(st, 0)
            self.assertIsInstance(st, int)

if __name__ == '__main__':
    unittest.main()

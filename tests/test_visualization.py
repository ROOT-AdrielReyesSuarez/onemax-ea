import os
import unittest
import numpy as np
from src.environment import DynamicOneMaxEnv
from src.optimizer import OnePlusOneEA
from src.simulation import ROOTSimulation
from src.visualization import plot_fitness_over_time, plot_survival_vs_kmax

class TestVisualizationAndMetrics(unittest.TestCase):
    def setUp(self):
        self.temp_fitness_file = "temp_fitness_test.png"
        self.temp_survival_file = "temp_survival_test.png"

    def tearDown(self):
        # Limpieza de archivos temporales de prueba si se crearon
        if os.path.exists(self.temp_fitness_file):
            os.remove(self.temp_fitness_file)
        if os.path.exists(self.temp_survival_file):
            os.remove(self.temp_survival_file)

    def test_plotting_exports_files(self):
        # 1. Verificar generación de imagen temporal de fitness
        dummy_history = [50, 60, 70, 80, 95, 95, 95, 85, 85, 75, 55]
        plot_fitness_over_time(dummy_history, acceptability_threshold=80, 
                               freeze_threshold=95, filename=self.temp_fitness_file)
        self.assertTrue(os.path.exists(self.temp_fitness_file), "El archivo temporal de fitness no se creó.")
        
        # 2. Verificar generación de imagen temporal de survival
        dummy_k_list = [2, 5, 10]
        dummy_survivals = [4.5, 2.1, 0.5]
        plot_survival_vs_kmax(dummy_k_list, dummy_survivals, filename=self.temp_survival_file)
        self.assertTrue(os.path.exists(self.temp_survival_file), "El archivo temporal de supervivencia no se creó.")

    def test_simulation_end_of_run_metric_recording(self):
        env = DynamicOneMaxEnv(length=20)
        optimizer = OnePlusOneEA(env=env)
        
        # Caso 1: Forzar una simulación con K_max muy bajo para provocar congelamiento prolongado
        sim = ROOTSimulation(
            env=env,
            optimizer=optimizer,
            acceptability_threshold=10,
            freeze_threshold=19,
            change_frequency=100,
            k_max=1  # Muy poca agresividad para asegurar supervivencia
        )
        
        # Corremos la simulación por solo 2 cambios
        sts = sim.run(target_env_changes=2, verbose=False)
        
        # Dado que K_max=1 y la frecuencia es 100, la solución debe congelarse rápidamente 
        # y sobrevivir casi con total certeza. Al final de la ejecución, debe registrarse
        # el Tiempo de Supervivencia actual acumulado.
        self.assertGreater(len(sts), 0, "No se registró ninguna métrica de supervivencia, incluso al finalizar congelado.")
        self.assertGreaterEqual(sts[-1], 0, "El último ST registrado no es válido.")

if __name__ == '__main__':
    unittest.main()

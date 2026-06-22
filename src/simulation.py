import numpy as np
from src.environment import DynamicOneMaxEnv
from src.optimizer import OnePlusOneEA

class ROOTSimulation:
    """
    Controlador para la simulación del paradigma ROOT (Robust Optimization Over Time).
    Gestiona el ciclo de optimización, congelamiento, perturbación ambiental
    y recolección de tiempos de supervivencia.
    """
    def __init__(self, env: DynamicOneMaxEnv, optimizer: OnePlusOneEA, 
                 acceptability_threshold: int = 80, freeze_threshold: int = 95,
                 change_frequency: int = 500, k_max: int = 15):
        self.env = env
        self.optimizer = optimizer
        self.mu = acceptability_threshold
        self.freeze_threshold = freeze_threshold
        self.change_frequency = change_frequency
        self.k_max = k_max
        
        # Historiales y métricas
        self.survival_times = []
        self.fitness_history = []
        self.env_changes = 0

    def run(self, target_env_changes: int = 20, verbose: bool = True) -> list[int]:
        """
        Ejecuta la simulación dinámica hasta alcanzar el número especificado de cambios ambientales.
        Retorna la lista de Tiempos de Supervivencia (ST).
        """
        n = self.env.length
        # Inicializar el padre de forma aleatoria
        parent = np.random.randint(2, size=n, dtype=np.int8)
        parent_fitness = self.env.get_fitness(parent)
        
        self.fitness_history = [parent_fitness]
        self.survival_times = []
        self.env_changes = 0
        
        current_st = 0
        is_frozen = False
        
        global_step = 0
        steps_since_last_change = 0
        
        if verbose:
            print("=== Inicio de la Simulación ROOT ===")
            print(f"Parámetros: mu={self.mu}, freeze_threshold={self.freeze_threshold}, Frecuencia={self.change_frequency}, K_max={self.k_max}\n")

        while self.env_changes < target_env_changes:
            global_step += 1
            steps_since_last_change += 1
            
            # 1. Comprobar si corresponde un cambio ambiental
            if steps_since_last_change >= self.change_frequency:
                self.env_changes += 1
                steps_since_last_change = 0
                
                # Sorteo de k ~ U(1, K_max)
                k = int(np.random.randint(1, self.k_max + 1))
                self.env.change_environment(k)
                
                # Evaluar la solución actual en el nuevo entorno
                parent_fitness = self.env.get_fitness(parent)
                
                if verbose:
                    if is_frozen:
                        if parent_fitness >= self.mu:
                            print(f"Entorno {self.env_changes:2d}: k={k:2d} sorteado -> Fitness={parent_fitness:2d} -> Sobrevive")
                        else:
                            print(f"Entorno {self.env_changes:2d}: k={k:2d} sorteado -> Fitness={parent_fitness:2d} -> MUERTE. Tiempo de Supervivencia registrado = {current_st}")
                    else:
                        print(f"Entorno {self.env_changes:2d}: k={k:2d} sorteado -> Fitness={parent_fitness:2d} -> No congelado (Optimizando...)")
                
                if is_frozen:
                    if parent_fitness >= self.mu:
                        # La solución sobrevive
                        current_st += 1
                    else:
                        # La solución muere, guardamos su ST y descongelamos para reparación
                        self.survival_times.append(current_st)
                        current_st = 0
                        is_frozen = False
                        
            # 2. Bucle de optimización / paso de evolución si no está congelado
            if not is_frozen:
                # Comprobar si la solución de partida ya cumple con el umbral de congelamiento
                if parent_fitness >= self.freeze_threshold:
                    is_frozen = True
                    current_st = 0
                    if verbose:
                        print(f"  [Congelamiento] Solución aceptable en paso {global_step} (Fitness={parent_fitness}). Congelando.")
                else:
                    # Mutación clásica del (1+1)-EA
                    child = self.optimizer.mutate(parent)
                    child_fitness = self.env.get_fitness(child)
                    
                    if child_fitness >= parent_fitness:
                        parent = child
                        parent_fitness = child_fitness
                        
                    if parent_fitness >= self.freeze_threshold:
                        is_frozen = True
                        current_st = 0
                        if verbose:
                            print(f"  [Congelamiento] Solución aceptable en paso {global_step} (Fitness={parent_fitness}). Congelando.")
            
            # Registrar fitness
            self.fitness_history.append(parent_fitness)
            
        if is_frozen:
            self.survival_times.append(current_st)
            
        return self.survival_times

if __name__ == '__main__':
    # Inicializar el entorno, optimizador y simulación para el Hito 3
    np.random.seed(42)  # Semilla fija para reproducibilidad del log
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
    
    sts = sim.run(target_env_changes=20, verbose=True)
    
    print("\n=== Resultados Finales ===")
    print(f"Tiempos de supervivencia registrados: {sts}")
    if len(sts) > 0:
        print(f"Tiempo de supervivencia promedio: {np.mean(sts):.2f} entornos")

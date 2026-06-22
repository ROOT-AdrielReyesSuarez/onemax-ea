import numpy as np
from src.environment import DynamicOneMaxEnv

class OnePlusOneEA:
    """
    Optimizador evolutivo (1+1)-EA para cadenas binarias.
    Mantiene un solo individuo (el Padre), genera un Hijo mediante mutación bit a bit
    y lo selecciona si su fitness es mayor o igual que el del Padre.
    """
    def __init__(self, env: DynamicOneMaxEnv, mutation_rate: float = None):
        self.env = env
        self.n = env.length
        # Tasa de mutación p = 1/n por defecto
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1.0 / self.n

    def mutate(self, parent: np.ndarray) -> np.ndarray:
        """
        Genera un hijo mutando cada bit del padre con una probabilidad independiente p.
        Utiliza vectorización de NumPy para máxima eficiencia.
        """
        if not isinstance(parent, np.ndarray):
            parent = np.array(parent, dtype=np.int8)
            
        child = parent.copy()
        
        # Generar máscara booleana con probabilidad independiente p para cada bit
        mutation_mask = np.random.rand(self.n) < self.mutation_rate
        
        # Invertir bits seleccionados usando XOR con 1
        child[mutation_mask] ^= 1
        
        return child

    def run_static(self, max_iterations: int = 1000) -> tuple[np.ndarray, list[int]]:
        """
        Ejecuta el ciclo de optimización (1+1)-EA clásico en un entorno estático.
        Retorna la mejor solución encontrada (el Padre final) y el historial de fitness.
        """
        # Generar padre inicial de forma aleatoria (0 o 1)
        parent = np.random.randint(2, size=self.n, dtype=np.int8)
        parent_fitness = self.env.get_fitness(parent)
        
        fitness_history = [parent_fitness]
        
        for _ in range(max_iterations):
            # Mutar
            child = self.mutate(parent)
            child_fitness = self.env.get_fitness(child)
            
            # Reemplazar si el hijo es mejor o igual
            if child_fitness >= parent_fitness:
                parent = child
                parent_fitness = child_fitness
                
            fitness_history.append(parent_fitness)
            
            # Condición de parada óptima: se alcanzó el óptimo absoluto
            if parent_fitness == self.n:
                break
                
        return parent, fitness_history

if __name__ == '__main__':
    print("=== Validación Aislada del Optimizador (Fase 2) ===")
    
    # 1. Crear entorno estático
    env = DynamicOneMaxEnv(length=100)
    print(f"Entorno estático inicializado. Longitud del problema (n): {env.length}")
    
    # 2. Inicializar el optimizador
    optimizer = OnePlusOneEA(env=env)
    print(f"Optimizador (1+1)-EA configurado con tasa de mutación p = {optimizer.mutation_rate:.4f} (1/n)\n")
    
    # 3. Ejecutar la optimización estática
    print("Iniciando ciclo de optimización estática (máx 1000 iteraciones)...")
    best_solution, history = optimizer.run_static(max_iterations=1000)
    
    # 4. Imprimir la evolución del fitness a intervalos regulares
    total_steps = len(history)
    print(f"\nResumen de Evolución:")
    print(f"  Paso   0 (Inicial): Fitness = {history[0]}")
    
    # Mostrar pasos intermedios (cada 100 generaciones)
    interval = 100
    for step in range(interval, total_steps, interval):
        print(f"  Paso {step:3d}:            Fitness = {history[step]}")
        
    # Mostrar el paso final
    if (total_steps - 1) % interval != 0:
        print(f"  Paso {total_steps-1:3d} (Final):   Fitness = {history[-1]}")
    
    print(f"\nMejor solución encontrada: {best_solution}")
    print(f"¿Alcanzó el óptimo global (100)? {'SÍ' if history[-1] == 100 else 'NO'}")
    
    # Asegurar que el algoritmo busca de forma eficiente y sube desde ~50
    assert history[0] < 75, f"Alerta: El fitness inicial es sospechosamente alto: {history[0]}"
    assert history[-1] >= 95, f"Alerta: El fitness final es menor del esperado: {history[-1]}"

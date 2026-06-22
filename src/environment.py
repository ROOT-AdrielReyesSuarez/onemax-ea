import numpy as np

class DynamicOneMaxEnv:
    """
    Entorno para el problema OneMax Dinámico y Variable.
    Representa el paisaje de fitness dinámico basado en una cadena óptima objetivo
    que sufre perturbaciones aleatorias a lo largo del tiempo.
    """
    def __init__(self, length: int = 100):
        if length <= 0:
            raise ValueError("La longitud del problema debe ser un entero positivo.")
        self.length = length
        # Generar el vector óptimo inicial aleatorio (0 o 1) con NumPy
        self.optimal_vector = np.random.randint(2, size=length, dtype=np.int8)

    def get_fitness(self, solution: np.ndarray) -> int:
        """
        Calcula el fitness de una solución candidata como la coincidencia posicional exacta
        (Distancia de Hamming invertida) con respecto al vector óptimo actual.
        """
        if not isinstance(solution, np.ndarray):
            solution = np.array(solution, dtype=np.int8)
        
        if solution.shape != self.optimal_vector.shape:
            raise ValueError(
                f"Dimensiones incompatibles: se esperaba un vector de tamaño {self.length}, "
                f"pero se recibió {solution.shape}."
            )
            
        # Comparación posicional exacta vectorizada con NumPy
        matches = np.sum(solution == self.optimal_vector)
        return int(matches)

    def change_environment(self, k: int) -> np.ndarray:
        """
        Perturba el entorno seleccionando exactamente k índices únicos al azar del vector óptimo
        e invirtiendo sus bits (0 -> 1, 1 -> 0).
        Retorna los índices que fueron modificados.
        """
        if not (0 <= k <= self.length):
            raise ValueError(f"El parámetro k ({k}) debe cumplir 0 <= k <= {self.length}.")
        
        if k == 0:
            return np.array([], dtype=np.intp)
            
        # Seleccionar k índices únicos de manera estrictamente aleatoria
        indices_to_flip = np.random.choice(self.length, size=k, replace=False)
        
        # Voltear los bits en esos índices usando XOR con 1
        self.optimal_vector[indices_to_flip] ^= 1
        
        return indices_to_flip

if __name__ == '__main__':
    # Bloque de ejecución directa para validación aislada del Hito 1
    print("=== Validación Aislada del Entorno (Fase 1) ===")
    
    # 1. Crear el entorno con n = 100
    n = 100
    env = DynamicOneMaxEnv(length=n)
    print(f"Entorno inicializado. Longitud del problema (n): {env.length}")
    print(f"Vector óptimo inicial: {env.optimal_vector}\n")
    
    # 2. Evaluar una solución idéntica al óptimo
    solucion_perfecta = env.optimal_vector.copy()
    fitness_perfecto = env.get_fitness(solucion_perfecta)
    print("--- Evaluación de Solución Perfecta ---")
    print(f"Solución candidata:   {solucion_perfecta}")
    print(f"Fitness obtenido:     {fitness_perfecto} (Esperado: 100)")
    assert fitness_perfecto == 100, f"Error: Fitness debería ser 100, pero es {fitness_perfecto}"
    
    # 3. Aplicar una perturbación manual controlada con un k fijo (ej. k = 15)
    k_fijo = 15
    print(f"\n--- Aplicando Perturbación (k = {k_fijo}) ---")
    indices_afectados = env.change_environment(k_fijo)
    print(f"Índices modificados ({len(indices_afectados)}): {sorted(indices_afectados.tolist())}")
    print(f"Nuevo vector óptimo:  {env.optimal_vector}")
    
    # 4. Verificar que el fitness de la solución original baja exactamente en la proporción esperada
    fitness_post_perturbacion = env.get_fitness(solucion_perfecta)
    print("\n--- Evaluación Post-Perturbación ---")
    print(f"Solución original evaluada en nuevo entorno: {solucion_perfecta}")
    print(f"Fitness obtenido:     {fitness_post_perturbacion} (Esperado: {100 - k_fijo})")
    
    diferencia = 100 - fitness_post_perturbacion
    print(f"Disminución de fitness: {diferencia} (Esperado: {k_fijo})")
    
    assert fitness_post_perturbacion == 100 - k_fijo, (
        f"Error: El fitness post-perturbación es {fitness_post_perturbacion}, "
        f"pero se esperaba {100 - k_fijo}."
    )
    print("\n¡Hito de Validación 1 superado con éxito!")

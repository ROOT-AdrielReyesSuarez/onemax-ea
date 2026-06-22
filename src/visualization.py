import numpy as np
import matplotlib.pyplot as plt
from src.environment import DynamicOneMaxEnv
from src.optimizer import OnePlusOneEA
from src.simulation import ROOTSimulation

def plot_fitness_over_time(history: list[int], acceptability_threshold: int = 80, 
                           freeze_threshold: int = 95, filename: str = "plots/fitness_over_time.png"):
    """
    Genera y guarda el gráfico de la evolución del fitness a lo largo de las iteraciones.
    Muestra el comportamiento en 'dientes de sierra', indicando las mesetas de estabilidad
    y las caídas causadas por los cambios de entorno.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history, label="Fitness del Padre", color="#1f77b4", linewidth=1.5)
    
    # Líneas discontinuas de los umbrales
    plt.axhline(y=acceptability_threshold, color="#d62728", linestyle="--", 
                linewidth=1.5, label=f"Umbral de Aceptabilidad Operativa (μ={acceptability_threshold})")
    plt.axhline(y=freeze_threshold, color="#2ca02c", linestyle="--", 
                linewidth=1.5, label=f"Umbral de Congelamiento (μ_freeze={freeze_threshold})")
    
    # Añadir relleno entre el umbral operativo y el de congelamiento para representar la zona de seguridad
    plt.axhspan(acceptability_threshold, freeze_threshold, color="#2ca02c", alpha=0.1, 
                label="Margen de Seguridad (15 bits)")
    
    plt.title("Evolución del Fitness en el Tiempo (Simulación ROOT)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Iteraciones Totales (Generaciones / Evaluaciones)", fontsize=12)
    plt.ylabel("Fitness (Coincidencia de bits con el óptimo)", fontsize=12)
    plt.ylim(35, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    import os
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Gráfico de fitness temporal guardado como: {filename}")

def plot_survival_vs_kmax(k_max_list: list[int], mean_survivals: list[float], 
                          filename: str = "plots/survival_vs_kmax.png"):
    """
    Genera y guarda el gráfico del Tiempo de Supervivencia Promedio vs. K_max.
    Muestra cómo la agresividad de las perturbaciones influye en la longevidad de las soluciones.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(k_max_list, mean_survivals, marker="o", color="#e377c2", 
             linewidth=2.5, markersize=8, label="Tiempo de Supervivencia (ST)")
    
    plt.title("Tiempo de Supervivencia Promedio vs. Agresividad Ambiental (K_max)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Límite de Agresividad Ambiental (K_max)", fontsize=12)
    plt.ylabel("Tiempo de Supervivencia Promedio (Nº de entornos sobrevividos)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(k_max_list)
    plt.legend(fontsize=10)
    plt.tight_layout()
    import os
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Gráfico de longevidad vs agresividad guardado como: {filename}")

def run_macro_experiment(k_max_list: list[int], runs_per_k: int = 10, 
                          target_changes: int = 20) -> list[float]:
    """
    Ejecuta simulaciones múltiples e independientes para diferentes valores de K_max 
    y recolecta el Tiempo de Supervivencia (ST) promedio para cada uno.
    """
    print("\nIniciando Experimento Macro: Tiempo de Supervivencia vs K_max")
    print(f"Configuración: {runs_per_k} ejecuciones independientes de {target_changes} cambios por cada K_max\n")
    
    mean_survivals = []
    
    for k_max in k_max_list:
        all_st = []
        for _ in range(runs_per_k):
            env = DynamicOneMaxEnv(length=100)
            optimizer = OnePlusOneEA(env=env)
            sim = ROOTSimulation(
                env=env,
                optimizer=optimizer,
                acceptability_threshold=80,
                freeze_threshold=95,
                change_frequency=500,
                k_max=k_max
            )
            # Simulación silenciosa para recolectar datos
            sts = sim.run(target_env_changes=target_changes, verbose=False)
            all_st.extend(sts)
            
        mean_st = np.mean(all_st) if len(all_st) > 0 else 0.0
        mean_survivals.append(mean_st)
        print(f"  K_max = {k_max:3d} | ST Promedio = {mean_st:6.2f} entornos (muestreo de {len(all_st)} soluciones)")
        
    return mean_survivals

if __name__ == '__main__':
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    # 1. Ejecutar una única simulación para capturar el historial de fitness en el tiempo (K_max = 15)
    print("=== Generando Datos para Gráfico Micro ===")
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
    sim.run(target_env_changes=20, verbose=False)
    plot_fitness_over_time(sim.fitness_history)
    
    # 2. Correr el experimento macro para obtener ST vs K_max
    k_max_list = [2, 5, 10, 20, 40, 60, 80, 100]
    mean_survivals = run_macro_experiment(k_max_list=k_max_list, runs_per_k=10, target_changes=20)
    plot_survival_vs_kmax(k_max_list, mean_survivals)
    
    print("\nVisualizaciones generadas correctamente.")

# ROOT-OneMax: Robust Optimization Over Time en Espacios Discretos

Este repositorio contiene el código experimental y la documentación para el estudio del paradigma de **Optimización Robusta a lo Largo del Tiempo (ROOT)** en espacios de búsqueda combinatorios binarios. Se utiliza como caso de estudio una variante dinámica del clásico problema **OneMax**, optimizado mediante un algoritmo evolutivo $(1+1)\text{-EA}$.

El objetivo principal de este proyecto es establecer un entorno de pruebas controlado (*baseline*) para medir el **Tiempo de Supervivencia** de las soluciones fijas frente a perturbaciones ambientales antes de abordar problemas de mayor complejidad.

---

## 1. Trasfondo Teórico: Paradigma ROOT vs. TMO

En la Optimización Dinámica tradicional, el enfoque predominante es el **Seguimiento del Óptimo Móvil** (*Tracking Moving Optimum* - TMO). TMO asume que el algoritmo debe reaccionar inmediatamente ante cualquier cambio ambiental, recalculando y desplegando la nueva solución óptima.

> [!WARNING]
> En escenarios del mundo real (procesos industriales, logística, redes de distribución), cambiar la solución física o de software de forma continua es inviable debido a los altos costes económicos de conmutación (*switching costs*), tiempos de reconfiguración o riesgos de inestabilidad operativa.

**ROOT (Robust Optimization Over Time)** propone una alternativa: en lugar de perseguir el óptimo ciegamente, busca soluciones que sean **suficientemente buenas (aceptables)** y que consigan **sobrevivir el mayor tiempo posible** frente a perturbaciones sin necesidad de ser modificadas o recalculadas.

| Característica | Enfoque TMO | Enfoque ROOT |
| :--- | :--- | :--- |
| **Objetivo** | Maximizar el fitness instantáneo (perseguir el óptimo). | Maximizar el tiempo de supervivencia de la solución fija. |
| **Acción ante cambios** | Re-optimizar y conmutar la solución de inmediato. | Mantener la solución congelada mientras sea aceptable. |
| **Sensibilidad** | Muy sensible a perturbaciones menores. | Muy tolerante, busca estabilidad a largo plazo. |
| **Costo Operativo** | Alto (debido a reconfiguraciones constantes). | Bajo (solución estable y estática). |

---

## 2. Especificación Técnica del Sistema

### A. El Entorno: OneMax Dinámico y Variable
* **Cadena Óptima Objetivo ( $O^*(t)$ ):** En $t=0$, el entorno genera un array de bits aleatorio de tamaño $n = 100$ que actúa como el óptimo actual.
* **Función de Fitness:** Mide la coincidencia posicional exacta con respecto al óptimo actual:
  $$f(X, t) = \sum_{i=1}^{n} [X[i] == O^*(t)[i]]$$
* **Perturbación Ambiental:** El entorno cambia de forma discreta cada $F = 500$ iteraciones. En cada cambio, se sortea una fuerza de agresividad $k \sim U(1, K_{max})$ y se invierte el valor de $k$ bits aleatorios del vector óptimo $O^*(t)$, alterando instantáneamente el paisaje de fitness.

### B. El Optimizador: $(1+1)\text{-EA}$ Vectorizado
1. **Población:** Mantiene un único individuo (Padre) representado por un array de 100 bits.
2. **Mutación:** El Hijo se genera copiando al Padre e invirtiendo cada bit con probabilidad independiente $p = 1/n$ (1% para $n=100$), implementado de forma vectorizada.
3. **Selección:** El Hijo reemplaza al Padre si su fitness es mayor o igual: $f(Hijo) \ge f(Padre)$.

### C. La Lógica de Doble Umbral y Margen de Seguridad

> [!IMPORTANT]
> Para evitar que las soluciones mueran inmediatamente ante el menor cambio del entorno, el sistema implementa una **estrategia de doble umbral** para introducir un margen de seguridad robusto del 15%:

1. **Umbral de Aceptabilidad Operativo ($\mu = 80$):** Si el fitness de la solución desplegada cae por debajo de 80, la solución muere, se detiene el registro de supervivencia y se activa el optimizador.
2. **Umbral de Congelamiento / Búsqueda ($\mu_{freeze} = 95$):** El optimizador no se detiene al cruzar el 80%, sino que sigue buscando de forma activa hasta alcanzar o superar un fitness de **95**. Una vez alcanzado, la solución se "congela" y se despliega fijamente.
3. **Métrica Principal: Tiempo de Supervivencia (ST):** Mide el número de cambios ambientales consecutivos que la solución fija es capaz de soportar manteniéndose por encima de $\mu \ge 80$.

---

## 3. Estructura del Proyecto

El código está organizado de manera modular, limpia y completamente probada:

```text
onemax-ea/
├── plots/                      # Carpeta de salida de gráficos
│   ├── fitness_over_time.png   # Comportamiento de fitness micro (dientes de sierra)
│   └── survival_vs_kmax.png    # Análisis macro de ST promedio vs K_max
├── src/                        # Código fuente
│   ├── environment.py          # Clase del entorno DynamicOneMaxEnv
│   ├── optimizer.py            # Clase del optimizador OnePlusOneEA
│   ├── simulation.py           # Ciclo dinámico e integrador de métricas
│   └── visualization.py        # Módulo de trazado y experimentos automatizados
├── tests/                      # Suite de pruebas unitarias
│   ├── test_environment.py     # Pruebas de paisaje de fitness y perturbaciones
│   ├── test_optimizer.py       # Pruebas de mutación y convergencia
│   ├── test_simulation.py      # Pruebas de máquina de estados ROOT
│   └── test_visualization.py   # Pruebas de integridad de archivos y exportación
└── requirements.txt            # Dependencias del proyecto (numpy, matplotlib)
```

---

## 4. Guía de Instalación y Preparación

Sigue estos pasos para clonar el repositorio y configurar el entorno de ejecución en tu máquina local:

### 1. Clonar el repositorio
Abre una terminal y clona este repositorio en tu directorio de trabajo:
```bash
git clone https://github.com/tu-usuario/onemax-ea.git
cd onemax-ea
```

### 2. Configurar el entorno virtual
Es muy recomendable aislar las dependencias utilizando un entorno virtual de Python (requiere Python 3.10 o superior):

* **En Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **En Windows (CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```
* **En macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar dependencias
Con el entorno virtual activo, instala las dependencias declaradas en el proyecto (NumPy y Matplotlib):
```bash
pip install -r requirements.txt
```

---

## 5. Instrucciones de Ejecución

El proyecto está diseñado para poder ejecutarse por componentes de forma directa utilizando el intérprete de Python del entorno virtual:

### Ejecutar Simulación Dinámica
Corre una simulación completa de 20 cambios de entorno con un log cronológico detallado en terminal:
```bash
python -m src.simulation
```

### Generar Gráficos de Resultados
Ejecuta la simulación micro y el experimento macro de análisis estadístico para generar los archivos gráficos PNG en la carpeta `plots/`:
```bash
python -m src.visualization
```
> [!NOTE]
> Una vez generado, puedes consultar `plots/fitness_over_time.png` para examinar el comportamiento en "dientes de sierra" del fitness y `plots/survival_vs_kmax.png` para ver el impacto de la agresividad ambiental en la longevidad de las soluciones.

### Ejecutar Suite de Pruebas Unitarias
Para correr las 10 pruebas unitarias que verifican la integridad algorítmica y del sistema:
```bash
python -m unittest discover -s tests
```

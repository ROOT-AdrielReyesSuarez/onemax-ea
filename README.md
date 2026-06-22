# ROOT-OneMax: Robust Optimization Over Time en Espacios Discretos

Este repositorio contiene el código base experimental para estudiar y evaluar el paradigma de **Optimización Robusta a lo Largo del Tiempo (ROOT)** sobre espacios de búsqueda combinatorios discretos (binarios), utilizando como caso de estudio una variante dinámica del clásico problema **OneMax**.

El objetivo principal de este proyecto es establecer un entorno de pruebas controlado (*baseline*) para medir el **Tiempo de Supervivencia** de las soluciones antes de abordar problemas abiertos en la literatura científica, tales como la simulación de perturbaciones ambientales asimétricas.

---

## 1. Trasfondo Teórico: Paradigma ROOT vs. TMO

En la Optimización Dinámica tradicional, el enfoque predominante es el **Seguimiento del Óptimo Móvil** (*Tracking Moving Optimum* - TMO). TMO asume que el algoritmo debe reaccionar inmediatamente a cualquier cambio del entorno para recalcular y perseguir el nuevo óptimo global. 

Sin embargo, en multitud de escenarios del mundo real (procesos industriales, logística, asignación de recursos), cambiar constantemente la solución desplegada es inviable debido a:
* Altos costes económicos de conmutación (*switching costs*).
* Tiempos físicos requeridos para reconfigurar el sistema.
* Inestabilidad operativa ante variaciones continuas.

**ROOT (Robust Optimization Over Time)** cambia radicalmente este enfoque: en lugar de perseguir el óptimo ciegamente, busca soluciones que sean **"lo suficientemente buenas" (aceptables)** y que consigan **sobrevivir el mayor tiempo posible** a lo largo de múltiples cambios ambientales sin necesidad de ser modificadas o recalculadas.

---

## 2. Definición del Entorno Dinámico: OneMax Dinámico

El problema clásico de *OneMax* consiste en encontrar una cadena de bits que maximice la cantidad de unos (`1`). En este proyecto, implementamos una variante **dinámica y variable en el espacio de estados**:

* **Cadena Óptima Objetivo ($O^*(t)$):** En el instante $t=0$, el entorno genera un array de bits aleatorio de tamaño $n = 100$ que actúa como la plantilla óptima actual.
* **Función de Aptitud (*Fitness*):** La calidad de una solución candidata $X$ no se mide por su número de unos absolutos, sino por la **coincidencia posicional exacta** con respecto a la cadena óptima del entorno:
  $$f(X, t) = \sum_{i=1}^{n} [X[i] == O^*(t)[i]]$$
* **Perturbación Ambiental (Parámetro $k$ aleatorio):** El entorno cambia de manera discreta cada cierto número fijo de evaluaciones. En cada cambio, el sistema determina una fuerza de cambio aleatoria controlada por un límite máximo ($k \in [1, K_{max}]$) y **voltea (*flips*) $k$ bits seleccionados al azar** de la cadena óptima $O^*(t)$, modificando instantáneamente el paisaje de fitness del problema.

---

## 3. Algoritmo de Optimización: $(1+1)\text{-EA}$ Discreto

Para la búsqueda de soluciones aceptables se utiliza un **Algoritmo Evolutivo $(1+1)\text{-EA}$**, caracterizado por su eficiencia y simplicidad estructural:

1. **Población:** Mantiene estrictamente un único individuo (el **Padre**), representado como un array de 100 bits.
2. **Mutación:** En cada iteración, se genera un único **Hijo** clonando al padre e invirtiendo cada uno de sus bits con una probabilidad estricta de mutación de $p = 1/n$ (lo que promedia una mutación de 1 bit por generación).
3. **Selección:** El hijo compite en un duelo directo contra el padre. Si el fitness del hijo es **mayor o igual** que el del padre frente al entorno actual, el hijo se convierte en el padre de la siguiente generación; de lo contrario, se descarta.

---

## 4. Criterio ROOT y Métricas Experimentales

El sistema opera bajo una estricta separación de fases (Optimización vs. Despliegue):

* **Umbral de Aceptabilidad ($\mu$):** Se fija una tolerancia del **80%**. Cualquier solución con un fitness $\ge 80$ (coincidencia en 80 o más bits con el óptimo actual) se considera **aceptable** para la operación.
* **Umbral de Congelamiento / Búsqueda ($\mu_{freeze}$):** Para dotar al sistema de un **margen de seguridad del 15%** contra perturbaciones, el proceso evolutivo no se detiene inmediatamente al cruzar el 80%, sino que continúa optimizando hasta alcanzar o superar un fitness del **95%** (95 bits). Una vez alcanzado, la solución se "despliega" de forma fija (congelamiento).
* **Métrica Principal: Tiempo de Supervivencia (*Survival Time* - ST):** Mide el número de cambios de entorno consecutivos que la solución fija es capaz de soportar manteniéndose por encima del umbral de aceptabilidad operativo ($\ge 80\%$).
* **Muerte y Reactivación:** Si tras un cambio ambiental (mutación de $k$ bits del óptimo) el fitness de la solución fija cae a 79 o menos, la solución muere. Se registra su Tiempo de Supervivencia y el algoritmo evolutivo se reactiva para encontrar una nueva solución que alcance nuevamente el umbral de congelamiento ($\ge 95\%$) en el entorno actual.

---

## 5. Objetivos de la Experimentación Visual

El software genera automáticamente dos tipos de análisis gráficos tras las simulaciones:
1. **Evolución del Fitness en el Tiempo:** Un gráfico temporal paso a paso que muestra el comportamiento en "dientes de sierra" del fitness, evidenciando las mesetas de estabilidad de la solución fija y las caídas verticales variables provocadas por los sorteos aleatorios de $k$.
2. **Tiempo de Supervivencia Promedio vs. Agresividad ($K_{max}$):** Un análisis macro que contrasta cómo influye el tope máximo de perturbación ambiental ($K_{max}$) en la esperanza de vida operativa de las soluciones generadas.
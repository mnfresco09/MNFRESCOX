# PANEL PORTFOLIO — Especificación de diseño

Fecha: 23 de junio de 2026  
Estado: aprobado para preparar el plan de implementación

## 1. Objetivo

Construir un módulo Python independiente para comparar siete métodos de
asignación de activos sobre cierres diarios:

1. Markowitz máximo Sharpe.
2. Markowitz con retorno objetivo.
3. Mínima varianza global.
4. Risk Parity por igualdad de contribuciones al riesgo.
5. Hierarchical Risk Parity (HRP).
6. Min-CVaR histórico.
7. Black-Litterman.

El panel debe producir asignaciones actuales, estimaciones de retorno y riesgo,
un backtest walk-forward fuera de muestra, análisis de cola y regímenes, stress
testing histórico y reportes HTML y Excel.

La implementación estará contenida íntegramente en `PANEL PORTFOLIO/`. No
importará ni leerá código, configuración o históricos de otros paneles.

## 2. Principios obligatorios

- Cada carpeta tiene una única responsabilidad.
- Las dependencias siguen un solo sentido y pasan por funciones públicas
  tipadas.
- `CONFIGURACION/` contiene exclusivamente parámetros.
- No se permiten datos inventados, correcciones silenciosas ni alternativas
  automáticas cuando falle una etapa.
- Los errores identifican etapa, método, activo o fecha y causa concreta.
- No se conserva código legacy ni se mantienen dos rutas para resolver el mismo
  problema.
- El código existente se audita: se conserva solo lo que cumpla este diseño y
  se reemplaza lo demás.
- Las métricas estimadas y los resultados históricos fuera de muestra se
  presentan por separado.

## 3. Estructura

```text
PANEL PORTFOLIO/
├── CONFIGURACION/   Parámetros editables por el usuario
├── CONTRATOS/       Dataclasses inmutables y validación de configuración
├── HISTORICO/       Parquet diarios propios
├── DESCARGADOR/     Descarga y validación de Yahoo Finance
├── DATOS/           Carga, validación, alineación y retornos
├── ANALISIS/        Estimadores, PCA, correlaciones y regímenes
├── OPTIMIZACION/    Siete asignadores, frontera y Monte Carlo
├── RIESGO/          Métricas, walk-forward, stress y análisis por régimen
├── REPORTES/        HTML autónomo, Excel y manifiesto
├── TESTS/           Pruebas unitarias, integración y regresión
├── ejecutar.py      Orquestador de comandos
└── requirements.txt Dependencias exclusivas del panel
```

`ejecutar.py` coordina las capas, pero no contiene lógica matemática. Ninguna
capa accederá a helpers privados de otra.

## 4. Configuración

`CONFIGURACION/config.py` será el único archivo que edita el usuario e incluirá:

- lista de tickers;
- fecha inicial y final;
- frecuencia de rebalanceo;
- ventana de estimación;
- restricción solo-largos;
- peso máximo por activo;
- retorno objetivo anual;
- tasa libre de riesgo;
- días de anualización;
- views opcionales de Black-Litterman;
- activo de referencia para cola y regímenes;
- coste de transacción en puntos básicos;
- nivel de confianza para VaR, CVaR y Min-CVaR, inicialmente `0.95`;
- número de carteras Monte Carlo y semilla;
- ventanas históricas de estrés;
- umbrales configurables de regímenes.

El activo de referencia debe pertenecer a la cesta. El valor inicial será
`^GSPC`.

Cuando `SOLO_LARGOS=True`, los pesos estarán entre cero y el máximo configurado.
Cuando sea `False`, `PESO_MAXIMO_POR_ACTIVO` actuará como límite absoluto:
`-máximo <= peso <= máximo`. Risk Parity y HRP serán siempre long-only porque
la extensión con cortos no corresponde a sus formulaciones estándar.

La configuración se validará antes de descargar o analizar. Entre otras
condiciones, se comprobarán fechas, tickers únicos, un mínimo de dos activos,
factibilidad de límites, confianza de views, ventana de estimación, nivel de
confianza de riesgo, costes y ventanas de estrés.

## 5. Contratos

Las entradas y salidas entre capas usarán dataclasses inmutables. Como mínimo:

- configuración validada;
- resumen de descarga por activo;
- conjunto de cierres;
- datos alineados y retornos;
- resultado de análisis;
- resultado de asignación;
- resultado de frontera y Monte Carlo;
- paso de rebalanceo;
- resultado walk-forward;
- métricas de cartera;
- resultado por régimen;
- resultado de stress;
- paquete final de reporte.

Los pesos conservarán siempre el orden explícito de los activos. Cada resultado
de optimización incluirá nombre, pesos, métricas estimadas, estado y diagnóstico
del solver y advertencias metodológicas aplicables.

## 6. Descarga e históricos

El comando:

```bash
python ejecutar.py descargar
```

realizará este flujo:

1. Validar la configuración.
2. Descargar cada activo desde Yahoo Finance con frecuencia diaria.
3. Preferir `Adj Close`; utilizar `Close` solo cuando el ajustado no exista.
4. Normalizar fechas sin zona horaria.
5. Validar esquema, orden, duplicados, nulos, precios no positivos, filas,
   cobertura real y huecos sospechosos.
6. Guardar inicialmente en una ubicación temporal.
7. Sustituir los Parquet de `HISTORICO/` solo cuando todos los activos sean
   válidos.
8. Imprimir un resumen con filas, fechas reales, huecos y archivo generado.

Si falla un activo, no se reemplazará ningún histórico existente. No se
continuará con una cesta parcial.

## 7. Preparación de datos

El comando:

```bash
python ejecutar.py analizar
```

usará exclusivamente los Parquet propios ya descargados.

`DATOS/`:

1. cargará y validará cada archivo;
2. comprobará cobertura y observaciones suficientes;
3. intersectará las fechas en las que cotizan todos los activos;
4. no aplicará `forward-fill`;
5. calculará log-retornos sobre los cierres alineados;
6. eliminará únicamente la primera fila resultante del cálculo de retornos;
7. verificará que no queden infinitos, nulos o columnas constantes.

La intersección evita fabricar retornos cero durante fines de semana o festivos
y mantiene consistentes covarianzas y correlaciones en cestas con calendarios
distintos.

El análisis estático exigirá al menos 252 retornos diarios alineados. El
walk-forward exigirá la ventana de estimación completa y al menos un periodo
posterior de aplicación. Si no existe esa cobertura, la ejecución se detendrá
indicando las observaciones requeridas y disponibles.

## 8. Análisis estadístico

`ANALISIS/` calculará:

- media histórica de log-retornos diarios y anualizada;
- covarianza Ledoit-Wolf diaria y anualizada;
- volatilidades;
- matriz de correlación media;
- matriz de correlación condicional sobre el peor decil de retornos del activo
  de referencia;
- diferencia entre correlación de cola y correlación media;
- PCA sobre retornos estandarizados, con varianza explicada y acumulada;
- diversification ratio;
- número efectivo de apuestas basado en contribuciones al riesgo;
- etiquetas diarias de régimen.

Los retornos esperados de Markowitz y Black-Litterman usarán la media histórica
de log-retornos alineados, anualizada con `DIAS_ANIO`. El reporte advertirá que
esta estimación tiene alta incertidumbre.

### 8.1 Regímenes

Las etiquetas serán transparentes y se calcularán solo con información
disponible hasta cada fecha:

- **Crisis:** drawdown del activo de referencia menor o igual a `-20 %`, o
  drawdown menor o igual a `-10 %` junto con volatilidad de 20 días
  excepcionalmente alta.
- **Bajista:** precio bajo la media de 200 días con pendiente negativa, o
  drawdown menor o igual a `-10 %`.
- **Alcista:** precio sobre la media de 200 días, pendiente positiva y drawdown
  superior a `-10 %`.
- **Lateral:** cualquier otro caso.

La volatilidad excepcional se determinará mediante un percentil expansivo
configurable, inicialmente el percentil 90. La pendiente será el cambio de la
media de 200 días durante las últimas 20 observaciones. Tanto el percentil como
la pendiente se calcularán sin usar observaciones futuras. Las observaciones
iniciales que no dispongan de 200 cierres se etiquetarán como `sin_clasificar`
y no se forzarán a otro régimen.

## 9. Optimización

Todos los métodos validarán que los pesos:

- sean finitos;
- sumen uno dentro de tolerancia;
- respeten las restricciones;
- produzcan métricas finitas.

Una optimización fallida detendrá la etapa. No se sustituirá por pesos iguales
ni por el resultado de otro método.

### 9.1 Markowitz máximo Sharpe

Optimización restringida sobre retorno esperado y covarianza Ledoit-Wolf. Se
reportará retorno esperado, volatilidad y Sharpe.

### 9.2 Markowitz retorno objetivo

Minimizará la varianza exigiendo el retorno anual configurado. Antes de
resolver, se calculará el rango de retorno alcanzable con las restricciones. Si
el objetivo no pertenece a ese rango, el análisis se detendrá indicando rango,
objetivo y restricción causante.

### 9.3 Mínima varianza global

Minimizará la varianza con suma de pesos igual a uno y las restricciones
configuradas.

### 9.4 Risk Parity

Igualará las contribuciones al riesgo mediante optimización. No se implementará
como `1/vol`. Será siempre long-only y respetará el peso máximo.

### 9.5 HRP

Implementará el flujo de López de Prado:

1. distancia derivada de correlaciones;
2. clustering jerárquico;
3. cuasi-diagonalización;
4. bisección recursiva por varianza de clúster.

No invertirá la matriz de covarianza. Será siempre long-only.

Si el resultado recursivo supera el peso máximo, se proyectará una sola vez
sobre el simplex acotado minimizando la distancia cuadrática respecto a los
pesos HRP originales. El reporte identificará que la restricción ha requerido
esta proyección. No se mantendrá una segunda variante de HRP.

### 9.6 Min-CVaR

Usará la formulación lineal de Rockafellar-Uryasev en `cvxpy`, tomando como
escenarios los retornos históricos reales. No asumirá normalidad.

### 9.7 Black-Litterman

El prior de mercado usará pesos iguales `1 / número de activos`, recalculados
para cada cesta. Sin views, devolverá limpiamente la cartera de equilibrio. Con
views, construirá las matrices correspondientes a opiniones absolutas o
relativas y aplicará la confianza configurada. Los retornos posteriores se
convertirán en pesos mediante una optimización de máximo Sharpe con la
covarianza posterior y las restricciones comunes.

### 9.8 Frontera y Monte Carlo

La frontera incluirá claramente:

- mínima varianza;
- máximo Sharpe;
- retorno objetivo.

La nube Monte Carlo respetará las restricciones. Si se permiten cortos, los
pesos se generarán dentro de los límites absolutos. Frontera y Monte Carlo son
análisis auxiliares, no métodos adicionales.

## 10. Walk-forward fuera de muestra

El motor generará rebalanceos mensuales —o con la frecuencia configurada— sobre
el calendario alineado.

En cada rebalanceo:

1. tomará exclusivamente las observaciones anteriores de la ventana de
   estimación;
2. recalculará análisis y pesos de los siete métodos;
3. aplicará los pesos a los retornos posteriores hasta el siguiente
   rebalanceo;
4. calculará rotación como la suma de cambios absolutos de peso;
5. descontará `rotación × COSTE_TRANSACCION_PB / 10 000`;
6. registrará pesos, rotación, coste, retornos OOS y diagnósticos.

Si un método falla en una fecha, se detendrá todo el backtest indicando método,
fecha y causa. No se rellenarán pesos.

Los estimadores trabajarán con log-retornos. Para simular la cartera, cada
log-retorno de activo se convertirá a retorno simple mediante `expm1`; el
retorno diario de cartera será el producto escalar de pesos y retornos simples,
y la curva de equity se compondrá multiplicativamente. El coste se descontará
en la primera observación OOS posterior a cada rebalanceo.

Este procedimiento impide valorar los métodos sobre los mismos datos usados
para estimarlos y reduce el espejismo in-sample que favorece especialmente a
Markowitz.

## 11. Riesgo y evaluación histórica

Las curvas OOS después de costes producirán:

- retorno anualizado;
- volatilidad anualizada;
- Sharpe;
- Sortino;
- Calmar;
- VaR histórico;
- CVaR o Expected Shortfall histórico;
- máximo drawdown;
- duración del drawdown;
- tiempo de recuperación cuando exista.

Las métricas por régimen se calcularán para cada método usando la etiqueta
contemporánea del activo de referencia. Se reportarán retorno, volatilidad,
drawdown y número de observaciones por régimen.

La diversificación media y en crisis incluirá diversification ratio y número
efectivo de apuestas.

## 12. Stress testing histórico

Las ventanas iniciales configurables serán:

- crisis financiera: 1 de septiembre de 2008 a 31 de marzo de 2009;
- COVID-19: 19 de febrero de 2020 a 23 de marzo de 2020;
- crisis de 2022: 3 de enero de 2022 a 12 de octubre de 2022.

Cada episodio se evaluará sobre las curvas walk-forward OOS. Una ventana se
recortará a la cobertura OOS disponible y se marcará como no evaluable cuando
contenga menos de cinco retornos comunes. No se considerará un error global que
una cesta o periodo no cubra un episodio.

## 13. Reportes

Cada análisis generará una carpeta de salida fechada con:

- HTML autónomo con Plotly embebido;
- libro Excel;
- manifiesto JSON de ejecución.

El HTML seguirá esta jerarquía:

1. resumen de configuración, cobertura, costes y advertencias;
2. asignación actual con pesos y métricas estimadas;
3. tabla separada de resultados walk-forward OOS después de costes;
4. frontera eficiente y Monte Carlo con los siete métodos;
5. curvas de equity OOS;
6. correlación media y condicional de cola;
7. PCA y diagnósticos de diversificación;
8. métricas por régimen;
9. stress testing histórico;
10. supuestos, restricciones, solver y diagnósticos.

El Excel tendrá hojas separadas para configuración, cobertura, pesos actuales,
métricas OOS, pesos walk-forward, rotación y costes, regímenes, stress,
correlaciones, PCA y diagnósticos.

El manifiesto JSON registrará parámetros, cobertura real, versiones de
dependencias y estado de cada etapa.

El reporte declarará expresamente que regímenes y episodios históricos son
análisis descriptivos del pasado y no predicen ni garantizan protección futura.

## 14. Tratamiento de errores

Se usarán excepciones específicas del panel con contexto estructurado. El
orquestador mostrará un mensaje conciso y finalizará con código distinto de
cero.

Casos que detienen la ejecución:

- configuración inválida;
- descarga vacía o corrupta;
- histórico ausente o inválido;
- cobertura común insuficiente;
- retornos no válidos;
- objetivo de retorno inviable;
- solver sin solución válida;
- pesos que incumplen restricciones;
- fallo al generar una salida requerida.

Las únicas ausencias no fatales serán ventanas de estrés sin cobertura y
regímenes sin observaciones suficientes; se reportarán como no evaluables.

## 15. Estrategia de pruebas

Las pruebas unitarias cubrirán:

- validación de configuración;
- guardado atómico y preservación de históricos ante fallo;
- alineación por intersección sin `forward-fill`;
- log-retornos;
- Ledoit-Wolf, PCA y correlación condicional;
- etiquetas de régimen sin información futura;
- restricciones y suma de pesos de cada optimizador;
- casos inviables;
- contribuciones de Risk Parity;
- ordenamiento y bisección HRP;
- formulación Min-CVaR;
- views Black-Litterman;
- frontera y Monte Carlo restringido;
- rebalanceos, rotación y costes;
- ausencia de look-ahead;
- VaR, CVaR, drawdown y recuperación;
- métricas por régimen y stress;
- HTML offline, Excel y manifiesto.

Una prueba de integración ejecutará el pipeline completo con datos sintéticos
deterministas. La verificación de la primera capa incluirá además una descarga
real con la cesta configurada antes de avanzar a análisis y optimización.

## 16. Orden de construcción y aceptación

La implementación se realizará por capas:

1. contratos, configuración, descargador y datos;
2. análisis estadístico;
3. siete optimizadores, uno a uno;
4. riesgo y walk-forward;
5. reportes.

No se avanzará de capa hasta que sus pruebas pasen y se hayan comprobado sus
salidas. Tras cada capa se comunicará qué se construyó, qué se verificó y el
resultado exacto.

La entrega se considerará completa cuando:

- ambos comandos funcionen desde `PANEL PORTFOLIO/`;
- la descarga real validada genere históricos propios;
- los siete métodos produzcan pesos válidos;
- el walk-forward sea estrictamente OOS y aplique costes;
- los análisis de cola, regímenes y stress estén presentes;
- HTML, Excel y manifiesto se generen correctamente;
- todas las pruebas pasen;
- no exista código duplicado, provisional o legacy.

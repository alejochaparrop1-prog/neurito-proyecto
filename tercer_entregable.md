Tercer entregable: Neurito en el laberinto

Estos son los experimentos de la guía y lo que salió en cada uno. 

Experimento 1: sin entrenar

¿Por qué se atasca? Porque todavía no sabe nada y se mueve casi al azar. Encontró la comida solo en el 3.1% de los retos. Casi siempre choca contra una pared y sigue intentando lo mismo, o se queda dando vueltas por el mismo camino.

Experimento 2: entrenar 100, 200 y 400 veces

Con 100 épocas, la pérdida fue 0.13, la precisión 86.3% y encontró la comida en el 61.1% de los retos.
Con 200 épocas, la pérdida fue 0.04, la precisión 90.1% y encontró la comida en el 81.2%.
Con 400 épocas, la pérdida fue 0.01, la precisión 91.1% y encontró la comida en el 83.4%.

Mientras más entrena, mejor le va, pero después de 200 épocas ya mejora poco. Con ayuda de las reglas también da menos vueltas: pasó de 18.2 movimientos sin entrenar a 6.8 con 400 épocas.

Experimento 3: red sola o con ayuda

La red sola encontró la comida el 83.4% de las veces. Las reglas solas, sin red, llegaron al 93.0%, pero dando muchas vueltas: 19.1 movimientos en promedio. Juntas llegaron al 100% con solo 6.8 movimientos.

La red aporta el saber hacia dónde ir, porque eso fue lo que aprendió. Las reglas evitan que choque con las paredes y que pase una y otra vez por el mismo lugar.

Experimento 4: cambiar las neuronas

Con 16 neuronas encontró la comida el 78.0% de las veces, con 32 el 83.1%, con 96 el 83.4% y con 128 el 83.2%. Con 16 le falta un poco, pero de 32 en adelante casi no cambia. Poner más neuronas no ayudó.

Experimento 5: cambiar la tasa de aprendizaje

Con 0.01 aprende muy lento y no alcanza a llegar: 20.0% de éxito.
Con 0.1 aprende, pero despacio: 62.7%.
Con 0.45, la de la guía, aprende bien: 83.4%.
Con 1.0 aprende más rápido: 85.1%.
Con 2.0 fue la mejor: 86.0%.
Con 5.0 es demasiado alta y no aprende: 4.8%.

Experimento 6: cambiar de mapa

Le fue parecido en los tres mapas: 84.2% en el primero, 81.8% en el segundo y 84.3% en el tercero. Con la ayuda de las reglas llegó al 100% en todos. Eso sí, los tres mapas ya los conocía de entrenar. Esto no prueba que funcione en un laberinto nuevo.

Experimento 7: DFS, BFS y A*

Le enseñé a Neurito con cada uno de los tres métodos.

Con DFS encontró la comida el 71.1% de las veces, y sus caminos fueron casi 3 veces más largos de lo necesario.
Con BFS encontró la comida el 83.4% de las veces, casi siempre por el camino más corto.
Con A* encontró la comida el 84.5% de las veces, también casi siempre por el camino más corto.

Con DFS empeora, porque ese método encuentra caminos largos y Neurito los copia. Entre BFS y A* casi no hay diferencia, porque los dos le enseñan caminos cortos.

Conclusión

Tener buena precisión no significa que Neurito termine el laberinto. Con 91.1% de precisión terminó el 83.4% de los retos, porque un solo error en el camino puede hacer que se atasque. Además, Neurito aprende copiando, así que lo hace tan bien como quien le enseña.

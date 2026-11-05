Segundo entregable: experimentos de Neurito 2.0

Aquí cuento lo que salió de los experimentos de la guía de Neurito. La idea era ver qué pasaba con la red cuando cambian el entrenamiento, las neuronas, la tasa de aprendizaje y la ventana de salto.

Cómo lo hice

No jugué las partidas a mano. Usé un script (experimentos_salto.py) que corre el juego sin ventana, con las mismas clases y las mismas reglas de saltar.py, porque así se pueden simular muchas partidas. Cada partida dura 9000 fotogramas, unos 5 minutos de juego. Entrené con 5000 ejemplos y probé con otros 1500. Casi siempre repetí todo con 5 semillas, para que el resultado no dependiera de la suerte de los pesos iniciales.

Cuando hablo de éxito me refiero al porcentaje de obstáculos esquivados entre todos los intentos.

Hay algo del juego que conviene tener en cuenta para leer los resultados: después de cada choque, la velocidad vuelve a 5.5. Si una red falla a esa velocidad, choca una y otra vez y nunca ve otra. Por eso un 0% de éxito no siempre quiere decir que falle a todas las velocidades. Para esos casos también probé un obstáculo a cada velocidad por separado, con la semilla 42 salvo donde digo otra cosa.

Experimento 1: sin entrenar

Con la red sin entrenar, Neurito no esquivó ni un obstáculo con ninguna de las 5 semillas. Con las semillas 42, 1 y 2 nunca saltó y chocó 59 veces en cada caso. Con la semilla 4 sí saltó, 120 veces, pero igual chocó 59 veces. Con la semilla 3 saltó 55 veces y chocó 55 veces.

Lo más raro fue la semilla 3. Sin haber entrenado tenía 89.1% de precisión, y aun así chocó siempre. Ahí ya se ve que una precisión alta no significa que juegue bien. Eso sí, al probarla por separado esquivó los obstáculos a velocidad 7.5 y 8.5, que en la partida nunca alcanzó a ver porque siempre fallaba a 5.5.

Experimento 2: entrenar de cien en cien

Aquí usé la red normal, con 12 neuronas, tasa 0.8 y semilla 42. A las 100 épocas la pérdida era 0.2601, la precisión 91.2% y Neurito chocó 5 veces. A las 200 la pérdida bajó a 0.2306, la precisión subió a 91.9% y solo chocó 1 vez. A las 300 ya no chocó más: esquivó los 74 obstáculos con una precisión de 92.3%.

Después siguió mejorando en los números. A las 1000 épocas la pérdida era 0.0574 y la precisión 97.8%. A las 10000 la pérdida era 0.0084 y la precisión 99.7%. En el juego no cambió nada, porque ya no chocaba. Esto lo medí con una sola semilla y una sola pista, así que no lo tomaría como regla general.

Experimento 3: cambiar las neuronas ocultas

Probé con 3, 12 y 24 neuronas, 5000 épocas y 5 semillas. Con 3 neuronas la red quedó claramente peor: 86.0% de precisión y 89.8% de éxito, y los resultados cambiaban mucho de una semilla a otra. Con 12 llegó a 99.5% de precisión y 100% de éxito. Con 24 llegó a 99.7% y también 100% de éxito, o sea que duplicar las neuronas casi no aportó nada.

En los tres casos la precisión con los datos de entrenamiento y con los de prueba fue casi igual, con una diferencia de 0.1 puntos o menos. Por eso no parece haber sobreajuste. Con 3 neuronas el problema no es que generalice mal, sino que le falta capacidad para aprender bien la regla.

Experimento 4: cambiar la tasa de aprendizaje

Probé las tasas 0.01, 0.8 y 1.2, con 5 semillas.

Con 0.01 aprende muy lento. A las 1000 épocas la pérdida seguía en 0.4111, y a las 20000 apenas llegaba a 0.2212, con 92.6% de precisión y 99.5% de éxito. Lo que más me llamó la atención fue que a las 1000 épocas tenía 87.5% de precisión y 0% de éxito. Con la semilla 42 saltaba muy pronto, unos 21 fotogramas antes de llegar al obstáculo. Eso sirve a velocidad 7.5 y 8.5, pero no a 5.5, así que se quedaba chocando a esa velocidad.

Con 0.8 fue la mejor combinación de velocidad y estabilidad. A las 5000 épocas tenía 99.5% de precisión y 100% de éxito, y a las 20000 la pérdida bajó hasta 0.0058.

Con 1.2 al principio aprende igual de rápido o un poco más: a las 5000 épocas la pérdida era 0.0136. Pero después se vuelve inestable, porque a las 20000 la pérdida promedio subió a 0.0455 y la precisión bajó de 99.7% a 99.0%. En el juego siguió con 100% de éxito.

También miré la curva de pérdida durante 5000 épocas con la semilla 42. Con 0.01 y con 0.8 nunca subió. Con 1.2 subió en 292 épocas, y la mayor subida fue de 0.0223, o sea que iba bajando a saltos. Para bajar de 0.30 de pérdida, la tasa 0.01 necesitó 3539 épocas, la 0.8 necesitó 46 y la 1.2 necesitó 38.

Experimento 5: cambiar la ventana de salto

Cambié la ventana en etiqueta() y en los ejemplos positivos, con 5000 épocas y 5 semillas. Antes medí qué permite la física. Si Neurito salta cuando faltan menos de 4 fotogramas, choca a cualquier velocidad. El límite por arriba depende de la velocidad: 14.8 fotogramas a velocidad 4.5, 18.0 a 5.5, 20.1 a 6.5, 21.7 a 7.5 y 22.9 a 8.5. Entonces, para que sirva a todas las velocidades, hay que saltar entre los 4 y los 14.8 fotogramas.

Con las ventanas 4-7 y 7-13 la red logró 100% de éxito, con 99.5% de precisión en las dos. Con 10-16 tuvo 99.3% de precisión y bajó a 93.6% de éxito, porque salta muy tarde para la velocidad 4.5, que es la más lenta. Con 13-20 tuvo 99.6% de precisión y 0% de éxito, aunque ese 0% se debe en parte al reinicio a 5.5: a velocidad 6.5, 7.5 y 8.5 sí esquiva. Con 20-30 tuvo 99.6% de precisión y con 7-25 tuvo 99.7%. Las dos dieron 0% de éxito, y ahí la red choca a todas las velocidades, porque salta demasiado pronto.

El caso 2-4 fue curioso. Tuvo 98.2% de precisión y 38.3% de éxito, con mucha diferencia entre semillas. La regla exacta, sin red, dio 0%, porque siempre salta cuando faltan menos de 4 fotogramas. La red a veces saltaba un poquito antes, con 4.1 o 4.4, y por eso se salvaba. Es decir, aquí le fue mejor por no aprender perfecto la regla.

Lo que saqué de este experimento es que Neurito salta la primera vez que la red dice que sí. Como el obstáculo viene de lejos, esa primera vez es casi siempre el borde alto de la ventana. Con 7-13 saltaba a una mediana de 12.9 fotogramas y con 7-25 a 24.4. El borde de abajo casi no importa. Por eso una ventana puede estar bien aprendida y aun así no ser compatible con la física.

Pregunta central: por qué una red puede clasificar bien y aun así chocar

La precisión cuenta cada ejemplo por separado, y todos valen lo mismo. En el juego no es así: basta un solo salto a destiempo para chocar, aunque casi todas las demás decisiones hayan estado bien.

Además, la precisión solo dice si la red aprendió la regla que le dimos, no si esa regla sirve para el juego. El mejor ejemplo es la ventana 7-25: la red aprendió con 99.7% de precisión a saltar en esa ventana, pero salta a unos 24 fotogramas del obstáculo y choca a todas las velocidades, porque el salto no alcanza. Las etiquetas nunca le dicen nada sobre la gravedad ni sobre cuánto dura el salto. La red no se equivocó: lo que estaba mal era la regla.

También importa en qué ejemplos se equivoca. La red sin entrenar de la semilla 3 tenía 89.1% de precisión y la de tasa 0.01 con 1000 épocas tenía 87.5%, y las dos sacaron 0% de éxito. En cambio, la red normal con 300 épocas tenía 92.3% y no chocó ninguna vez. Son pocos puntos de diferencia y un resultado totalmente distinto.

Y el resultado depende también de cómo funciona el juego. Con la ventana 13-20, la red esquiva a velocidad 6.5, 7.5 y 8.5, pero el reinicio a 5.5 la deja atrapada en la velocidad donde falla. Por eso para evaluar a Neurito no basta con la precisión: hay que ponerlo a jugar.

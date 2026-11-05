"""Experimentos de GUIA_SALTO.md sin ventana, con la misma lógica de saltar.py.

Uso: uv run experimentos_salto.py
No modifica salto_modelo.py ni salto_juego.py: los cambios de cada experimento
se hacen con parámetros del constructor o con una subclase.
"""

import random

import numpy as np

from salto_juego import JuegoSalto
from salto_modelo import RedSalto

SEMILLAS = [42, 1, 2, 3, 4]  # 42 es la que usa saltar.py
FRAMES = 9000  # 5 minutos de juego a 30 fotogramas por segundo
PISTAS = [19, 7]  # 19 es la pista de saltar.py


class RedVentana(RedSalto):
    """Experimento 5: cambia la ventana en etiqueta() y en los positivos."""

    def __init__(self, ventana, **kwargs):
        self.ventana = ventana
        super().__init__(**kwargs)

    def etiqueta(self, distancia, velocidad):
        ttc = np.asarray(distancia) / np.maximum(np.asarray(velocidad), 0.001)
        lo, hi = self.ventana
        return ((ttc >= lo) & (ttc <= hi)).astype(float).reshape(-1, 1)

    def generar_datos(self, n=5000):
        lo, hi = self.ventana
        v = self.rng.uniform(4.5, 8.5, n)
        n_pos = n // 2
        ttc = np.concatenate((
            self.rng.uniform(lo + 0.5, hi - 0.5, n_pos),
            self.rng.uniform(0, 60, n - n_pos),
        ))
        self.rng.shuffle(ttc)
        d = v * ttc
        return self.preparar_entradas(d, v), self.etiqueta(d, v)


def crear(semilla=42, ocultas=12, tasa=0.8, ventana=None):
    """Mismo orden de creación que saltar.py: pesos, entrenamiento y prueba."""
    if ventana is None:
        red = RedSalto(semilla=semilla, ocultas=ocultas, tasa=tasa)
    else:
        red = RedVentana(ventana, semilla=semilla, ocultas=ocultas, tasa=tasa)
    X, y = red.generar_datos(5000)
    X_test, y_test = red.generar_datos(1500)
    return red, X, y, X_test, y_test


def entrenar_hasta(red, X, y, epocas):
    while red.epocas < epocas:
        red.entrenar_epoca(X, y)


def precision_frontera(red, n=4000):
    """Precisión solo donde la decisión es difícil: tiempo estimado de 5 a 15."""
    rng = np.random.default_rng(123)
    v = rng.uniform(4.5, 8.5, n)
    d = v * rng.uniform(5, 15, n)
    return red.precision(red.preparar_entradas(d, v), red.etiqueta(d, v))


def jugar(politica, frames=FRAMES, pista=19):
    """Bucle de saltar.py: decidir, saltar si está en el suelo y actualizar."""
    juego = JuegoSalto()
    juego.azar = random.Random(pista)
    saltos = []
    causas = {"sin_saltar": 0, "subiendo": 0, "bajando": 0}
    aciertos_frame = total_frame = 0
    for _ in range(frames):
        if not juego.terminado:
            d, v = juego.distancia, juego.velocidad
            quiere = politica(d, v)
            if juego.en_suelo:
                total_frame += 1
                aciertos_frame += quiere == (7 <= d / v <= 13)
                if quiere:
                    saltos.append(d / v)
                    juego.saltar()
        antes = juego.choques
        juego.actualizar()
        if juego.choques > antes:
            if juego.en_suelo:
                causas["sin_saltar"] += 1
            elif juego.vy < 0:
                causas["subiendo"] += 1
            else:
                causas["bajando"] += 1
    total = juego.esquivados + juego.choques
    return {
        "esquivados": juego.esquivados,
        "choques": juego.choques,
        "exito": juego.esquivados / total if total else 0.0,
        "saltos": len(saltos),
        "causas": causas,
        "ttc_saltos": saltos,
        "precision_frames": aciertos_frame / max(total_frame, 1),
    }


def politica_red(red):
    return lambda d, v: red.probabilidad(d, v) >= 0.5


def exito_medio(red):
    resultados = [jugar(politica_red(red), pista=p) for p in PISTAS]
    return float(np.mean([r["exito"] for r in resultados])), resultados


def media_std(valores, pct=True):
    k = 100 if pct else 1
    formato = "{:.1f}% ± {:.1f}" if pct else "{:.4f} ± {:.4f}"
    return formato.format(np.mean(valores) * k, np.std(valores) * k)


def tabla(cabecera, filas):
    print("| " + " | ".join(cabecera) + " |")
    print("|" + "---|" * len(cabecera))
    for fila in filas:
        print("| " + " | ".join(str(c) for c in fila) + " |")
    print()


def experimento_1():
    print("## Experimento 1: sin entrenar\n")
    filas = []
    for s in SEMILLAS:
        red, _, _, X_t, y_t = crear(s)
        r = jugar(politica_red(red))
        c = r["causas"]
        filas.append((s, f"{red.precision(X_t, y_t):.1%}", r["esquivados"], r["choques"],
                      f"{r['exito']:.1%}", r["saltos"],
                      f"{c['sin_saltar']}/{c['subiendo']}/{c['bajando']}"))
    tabla(("semilla", "precisión", "esquivados", "choques", "éxito", "saltos",
           "choques suelo/subiendo/bajando"), filas)


def experimento_2():
    print("## Experimento 2: de cien en cien épocas (semilla 42, pista 19)\n")
    red, X, y, X_t, y_t = crear(42)
    filas = []
    for ep in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]:
        entrenar_hasta(red, X, y, ep)
        r = jugar(politica_red(red))
        c = r["causas"]
        filas.append((ep, f"{red.historial[-1]:.4f}", f"{red.precision(X_t, y_t):.1%}",
                      r["esquivados"], r["choques"], f"{r['exito']:.1%}",
                      f"{c['sin_saltar']}/{c['subiendo']}/{c['bajando']}"))
    tabla(("épocas", "pérdida", "precisión prueba", "esquivados", "choques", "éxito",
           "choques suelo/subiendo/bajando"), filas)


def experimento_3(epocas=5000):
    print(f"## Experimento 3: neuronas ocultas ({epocas} épocas, media de 5 semillas)\n")
    filas = []
    for ocultas in [3, 12, 24]:
        ent, pru, fro, jue, per = [], [], [], [], []
        for s in SEMILLAS:
            red, X, y, X_t, y_t = crear(s, ocultas=ocultas)
            entrenar_hasta(red, X, y, epocas)
            per.append(red.historial[-1])
            ent.append(red.precision(X, y))
            pru.append(red.precision(X_t, y_t))
            fro.append(precision_frontera(red))
            jue.append(exito_medio(red)[0])
        filas.append((ocultas, media_std(per, pct=False), media_std(ent), media_std(pru),
                      f"{(np.mean(ent) - np.mean(pru)) * 100:+.1f}", media_std(fro),
                      media_std(jue)))
    tabla(("ocultas", "pérdida", "precisión entren.", "precisión prueba",
           "brecha (puntos)", "precisión frontera", "éxito en juego"), filas)


def experimento_4():
    print("## Experimento 4: tasa de aprendizaje (media de 5 semillas)\n")
    filas = []
    for tasa in [0.01, 0.8, 1.2]:
        for ep in [100, 1000, 5000, 20000]:
            per, pru, jue = [], [], []
            for s in SEMILLAS:
                red, X, y, X_t, y_t = crear(s, tasa=tasa)
                entrenar_hasta(red, X, y, ep)
                per.append(red.historial[-1])
                pru.append(red.precision(X_t, y_t))
                jue.append(exito_medio(red)[0])
            filas.append((tasa, ep, media_std(per, pct=False), media_std(pru), media_std(jue)))
    tabla(("tasa", "épocas", "pérdida", "precisión prueba", "éxito en juego"), filas)

    print("Estabilidad de la curva de pérdida en 5000 épocas (semilla 42):\n")
    filas = []
    for tasa in [0.01, 0.8, 1.2]:
        red, X, y, _, _ = crear(42, tasa=tasa)
        entrenar_hasta(red, X, y, 5000)
        h = np.array(red.historial)
        difs = np.diff(h)
        bajo = np.nonzero(h < 0.30)[0]
        filas.append((tasa, f"{h[0]:.4f}", f"{h[-1]:.4f}", int(np.sum(difs > 0)),
                      f"{difs.max():.4f}", int(bajo[0]) + 1 if len(bajo) else "nunca"))
    tabla(("tasa", "pérdida inicial", "pérdida final", "épocas en que sube",
           "mayor subida", "primera época con pérdida < 0.30"), filas)


def mapa_fisico():
    print("## Física: tiempo estimado de salto que esquiva el obstáculo\n")
    filas = []
    for v in [4.5, 5.5, 6.5, 7.5, 8.5]:
        buenos = []
        for t10 in range(0, 601):
            t = t10 / 10
            juego = JuegoSalto()
            juego.velocidad = v
            juego.obstaculo_x = 168 + v * t
            juego.saltar()
            while juego.esquivados == 0 and not juego.terminado:
                juego.actualizar()
            if juego.esquivados:
                buenos.append(t)
        filas.append((v, f"{min(buenos):.1f}", f"{max(buenos):.1f}"))
    tabla(("velocidad", "tiempo mínimo", "tiempo máximo"), filas)


def experimento_5(epocas=5000):
    print(f"## Experimento 5: ventana de entrenamiento ({epocas} épocas, media de 5 semillas)\n")
    filas = []
    for ventana in [(2, 4), (4, 7), (7, 13), (10, 16), (13, 20), (20, 30), (7, 25)]:
        lo, hi = ventana

        def regla(d, v, lo=lo, hi=hi):
            return lo <= d / v <= hi

        regla_r = [jugar(regla, pista=p)["exito"] for p in PISTAS]
        pru, jue, ttc = [], [], []
        for s in SEMILLAS:
            red, X, y, X_t, y_t = crear(s, ventana=ventana)
            entrenar_hasta(red, X, y, epocas)
            pru.append(red.precision(X_t, y_t))
            exito, rs = exito_medio(red)
            jue.append(exito)
            ttc += [t for r in rs for t in r["ttc_saltos"]]
        filas.append((f"{lo}-{hi}", media_std(pru), f"{np.mean(regla_r):.1%}", media_std(jue),
                      f"{np.median(ttc):.1f}" if ttc else "-"))
    tabla(("ventana", "precisión prueba", "éxito de la regla exacta", "éxito de la red",
           "mediana del tiempo al saltar"), filas)


def pregunta_central():
    print("## Pregunta central: red de referencia, 5000 épocas (semilla 42, pista 19)\n")
    red, X, y, X_t, y_t = crear(42)
    entrenar_hasta(red, X, y, 5000)
    r = jugar(politica_red(red))
    p = np.percentile(r["ttc_saltos"], [0, 5, 50, 95, 100])
    print(f"- Precisión de prueba: {red.precision(X_t, y_t):.1%}")
    print(f"- Precisión en fotogramas reales sobre el suelo: {r['precision_frames']:.1%}")
    print(f"- Éxito en el juego: {r['exito']:.1%}; causas: {r['causas']}")
    print(f"- Tiempo al saltar (mín, p5, mediana, p95, máx): {np.round(p, 1).tolist()}")
    print(f"- Saltos por obstáculo: {r['saltos'] / max(r['esquivados'] + r['choques'], 1):.2f}\n")


def un_obstaculo(politica, v):
    """Un obstáculo a velocidad fija, sin el reinicio a 5.5 que sigue a un choque."""
    juego = JuegoSalto()
    juego.velocidad = v
    tiempo = "sin salto"
    while not juego.esquivados and not juego.terminado:
        d = juego.distancia
        if juego.en_suelo and politica(d, v):
            tiempo = f"{d / v:.1f}"
            juego.saltar()
        juego.actualizar()
    return f"{'esquiva' if juego.esquivados else 'choca'} ({tiempo})"


def prueba_por_velocidad():
    print("## Resultado por velocidad (semilla 42; entre paréntesis, el tiempo al saltar)\n")
    velocidades = [4.5, 5.5, 6.5, 7.5, 8.5]
    casos = []
    for semilla in [42, 3, 4]:
        casos.append((f"sin entrenar, semilla {semilla}", crear(semilla)[0]))
    red, X, y, _, _ = crear(42, tasa=0.01)
    entrenar_hasta(red, X, y, 1000)
    casos.append(("tasa 0.01, 1000 épocas", red))
    for ventana in [(2, 4), (7, 13), (10, 16), (13, 20), (20, 30), (7, 25)]:
        red, X, y, _, _ = crear(42, ventana=ventana)
        entrenar_hasta(red, X, y, 5000)
        casos.append((f"ventana {ventana[0]}-{ventana[1]}", red))
    filas = [(nombre, *(un_obstaculo(politica_red(red), v) for v in velocidades))
             for nombre, red in casos]
    tabla(("caso", *(f"v={v}" for v in velocidades)), filas)


if __name__ == "__main__":
    experimento_1()
    experimento_2()
    experimento_3()
    experimento_4()
    mapa_fisico()
    experimento_5()
    prueba_por_velocidad()
    pregunta_central()

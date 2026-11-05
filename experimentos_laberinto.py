"""Experimentos de GUIA_LABERINTO.md sin ventana, con las reglas de laberinto.py.

Uso: uv run experimentos_laberinto.py
No modifica los archivos del laberinto: DFS y A* se definen aquí como profesores
alternativos para el experimento 7.
"""
import heapq

import numpy as np

from laberinto_mundo import (
    Laberinto, MAPAS, accion_experta, distancias_bfs, elegir_accion, libres,
    valido, vecino,
)
from laberinto_red import RedLaberinto, codificar, generar_datos

SEMILLAS = [42, 1, 2, 3, 4]  # 42 es la que usa laberinto.py
LIMITE_INTENTOS = 45  # laberinto.py declara atasco al llegar a 45 intentos
np.seterr(all="ignore")  # con tasas altas los pesos pueden desbordarse


# ---------------------------------- Evaluación ---------------------------------

class SoloReglas:
    """Sin modelo: todas las salidas valen igual, solo actúan las reglas asistidas."""

    @staticmethod
    def probabilidades(indice, posicion, meta):
        return np.full(4, 0.25)


def recorrer(decidir, indice, inicio, meta):
    """Un reto completo con Laberinto.mover, como en el bucle de laberinto.py."""
    mundo = Laberinto(indice)
    mundo.posicion, mundo.meta = inicio, meta
    mundo.visitadas = {inicio: 1}
    while True:
        resultado = mundo.mover(decidir(mundo))
        if resultado == "comida":
            return "exito", mundo.pasos
        if mundo.intentos >= LIMITE_INTENTOS:
            return ("pared" if resultado == "pared" else "ciclo"), mundo.pasos


def evaluar(decidir, casos):
    optimos = {}
    exitos, pasos, razones = 0, [], []
    fallos = {"pared": 0, "ciclo": 0}
    for indice, inicio, meta, *_ in casos:
        if (indice, meta) not in optimos:
            optimos[(indice, meta)] = distancias_bfs(MAPAS[indice], meta)
        final, n = recorrer(decidir, indice, inicio, meta)
        if final == "exito":
            exitos += 1
            pasos.append(n)
            razones.append(n / optimos[(indice, meta)][inicio])
        else:
            fallos[final] += 1
    return {
        "exito": exitos / len(casos),
        "pasos": float(np.mean(pasos)) if pasos else float("nan"),
        "razon": float(np.mean(razones)) if razones else float("nan"),
        **fallos,
    }


def con_red(red, asistido):
    return lambda mundo: elegir_accion(red, mundo, mundo.visitadas, asistido)[0]


def media_std(valores, pct=True):
    k = 100 if pct else 1
    formato = "{:.1f}% ± {:.1f}" if pct else "{:.4f} ± {:.4f}"
    return formato.format(np.mean(valores) * k, np.std(valores) * k)


def tabla(cabecera, filas):
    print("| " + " | ".join(cabecera) + " |")
    print("|" + "---|" * len(cabecera))
    for fila in filas:
        print("| " + " | ".join(str(c) for c in fila) + " |")
    print(flush=True)


def resumen_redes(datos, casos, epocas=400, **kwargs):
    """Entrena una red por semilla y devuelve las medidas de cada una."""
    (X, y), (Xt, yt) = datos
    salida = []
    for s in SEMILLAS:
        red = RedLaberinto(semilla=s, **kwargs)
        for _ in range(epocas):
            red.entrenar_epoca(X, y)
        pura = evaluar(con_red(red, False), casos)
        asistida = evaluar(con_red(red, True), casos)
        salida.append({
            "red": red,
            "perdida": red.historial[-1] if red.historial else float("nan"),
            "ent": red.exactitud(X, y),
            "pru": red.exactitud(Xt, yt),
            "pura": pura,
            "asistida": asistida,
        })
    return salida


def col(rs, *clave, pct=True):
    valores = []
    for r in rs:
        v = r
        for k in clave:
            v = v[k]
        valores.append(v)
    return media_std(valores, pct=pct)


def media(rs, *clave):
    valores = []
    for r in rs:
        v = r
        for k in clave:
            v = v[k]
        valores.append(v)
    return float(np.nanmean(valores))


# --------------------------------- Experimentos --------------------------------

def experimento_1(datos, casos):
    print("## Experimento 1: sin entrenar (retos de prueba)\n")
    filas = []
    for r, s in zip(resumen_redes(datos, casos, epocas=0), SEMILLAS):
        p, a = r["pura"], r["asistida"]
        filas.append((s, f"{r['pru']:.1%}", f"{p['exito']:.1%}", p["pared"], p["ciclo"],
                      f"{a['exito']:.1%}", a["pared"], a["ciclo"]))
    tabla(("semilla", "exactitud", "éxito pura", "atascos pared", "atascos ciclo",
           "éxito asistida", "asistida pared", "asistida ciclo"), filas)


def experimento_2_3(datos, casos):
    print("## Experimentos 2 y 3: épocas y modos (media de 5 semillas)\n")
    filas = []
    for ep in [0, 100, 200, 400]:
        rs = resumen_redes(datos, casos, epocas=ep)
        filas.append((ep, f"{media(rs, 'perdida'):.4f}", col(rs, "pru"),
                      col(rs, "pura", "exito"), f"{media(rs, 'pura', 'pasos'):.1f}",
                      f"{media(rs, 'pura', 'pared'):.0f}/{media(rs, 'pura', 'ciclo'):.0f}",
                      col(rs, "asistida", "exito"), f"{media(rs, 'asistida', 'pasos'):.1f}",
                      f"{media(rs, 'asistida', 'razon'):.2f}"))
    tabla(("épocas", "pérdida", "exactitud prueba", "éxito pura", "pasos pura",
           "fallos pura pared/ciclo", "éxito asistida", "pasos asistida",
           "pasos asistida / óptimo"), filas)
    reglas = evaluar(con_red(SoloReglas, True), casos)
    print(f"Solo reglas, sin modelo: éxito {reglas['exito']:.1%}, "
          f"pasos {reglas['pasos']:.1f}, pasos/óptimo {reglas['razon']:.2f}, "
          f"fallos pared/ciclo {reglas['pared']}/{reglas['ciclo']}\n", flush=True)


def experimento_4(datos, casos):
    print("## Experimento 4: neuronas ocultas (400 épocas, media de 5 semillas)\n")
    filas = []
    for ocultas in [16, 32, 96, 128]:
        rs = resumen_redes(datos, casos, ocultas=ocultas)
        filas.append((ocultas, f"{media(rs, 'perdida'):.4f}", col(rs, "ent"), col(rs, "pru"),
                      f"{(media(rs, 'ent') - media(rs, 'pru')) * 100:.1f}",
                      col(rs, "pura", "exito"), col(rs, "asistida", "exito")))
    tabla(("ocultas", "pérdida", "exactitud entren.", "exactitud prueba", "brecha (puntos)",
           "éxito pura", "éxito asistida"), filas)


def experimento_5(datos, casos):
    print("## Experimento 5: tasa de aprendizaje (400 épocas, media de 5 semillas)\n")
    filas = []
    for tasa in [0.01, 0.1, 0.45, 1.0, 2.0, 5.0]:
        rs = resumen_redes(datos, casos, tasa=tasa)
        hist = [np.array(r["red"].historial) for r in rs]
        subidas = np.mean([np.sum(np.diff(h) > 0) for h in hist])
        primera = [int(np.argmax(h < 0.5)) + 1 if np.any(h < 0.5) else None for h in hist]
        llegan = [p for p in primera if p is not None]
        filas.append((tasa, f"{np.mean([h[0] for h in hist]):.3f}", f"{media(rs, 'perdida'):.4f}",
                      f"{np.mean([h.min() for h in hist]):.4f}", f"{subidas:.0f}",
                      f"{np.mean(llegan):.0f} ({len(llegan)}/5)" if llegan else "nunca",
                      col(rs, "pru"), col(rs, "pura", "exito"), col(rs, "asistida", "exito")))
    tabla(("tasa", "pérdida época 1", "pérdida final", "pérdida mínima", "épocas en que sube",
           "primera época < 0.5", "exactitud prueba", "éxito pura", "éxito asistida"), filas)


def experimento_6(datos, muestras_ent, casos):
    print("## Experimento 6: por mapa (400 épocas, media de 5 semillas)\n")
    rs = resumen_redes(datos, casos[:1])
    filas = []
    for indice in range(len(MAPAS)):
        pru = [c for c in casos if c[0] == indice]
        ent = [c for c in muestras_ent if c[0] == indice]
        Xp = np.stack([codificar(*c[:3]) for c in pru])
        yp = np.array([c[3] for c in pru])
        exact, pura_p, asist_p, pura_e = [], [], [], []
        for r in rs:
            red = r["red"]
            exact.append(red.exactitud(Xp, yp))
            pura_p.append(evaluar(con_red(red, False), pru)["exito"])
            asist_p.append(evaluar(con_red(red, True), pru)["exito"])
            pura_e.append(evaluar(con_red(red, False), ent)["exito"])
        filas.append((indice + 1, len(ent), len(pru), media_std(exact), media_std(pura_e),
                      media_std(pura_p), media_std(asist_p)))
    tabla(("mapa", "pares entren.", "pares prueba", "exactitud prueba",
           "éxito pura en pares de entren.", "éxito pura en pares de prueba",
           "éxito asistida en pares de prueba"), filas)
    pares_ent_mapa = {(c[0], c[1], c[2]) for c in muestras_ent}
    pares_ent = {(c[1], c[2]) for c in muestras_ent}
    print(f"- Pares de prueba repetidos en el mismo mapa: "
          f"{sum((c[0], c[1], c[2]) in pares_ent_mapa for c in casos)} de {len(casos)}")
    print(f"- Pares de prueba cuya posición y meta sí aparecen en entrenamiento en otro mapa: "
          f"{sum((c[1], c[2]) in pares_ent for c in casos)} de {len(casos)}\n", flush=True)


# ---------------------------- Experimento 7: profesores -------------------------

def accion_hacia(origen, destino):
    return next(a for a in range(4) if vecino(origen, a) == destino)


def arbol_dfs(mapa, meta):
    """DFS desde la meta: seguir al padre siempre llega, igual que BFS desde la meta."""
    padre = {meta: None}

    def visitar(celda):
        for a in range(4):
            siguiente = vecino(celda, a)
            if valido(mapa, siguiente) and siguiente not in padre:
                padre[siguiente] = celda
                visitar(siguiente)

    visitar(meta)
    return padre


def a_estrella(mapa, origen, meta):
    """A* con distancia Manhattan. Devuelve primer paso, longitud y nodos expandidos."""
    def h(p):
        return abs(p[0] - meta[0]) + abs(p[1] - meta[1])

    contador = 0
    abiertos = [(h(origen), h(origen), contador, origen, 0, None)]
    cerrados = set()
    expandidos = 0
    while abiertos:
        _, _, _, actual, g, primero = heapq.heappop(abiertos)
        if actual in cerrados:
            continue
        cerrados.add(actual)
        expandidos += 1
        if actual == meta:
            return primero, g, expandidos
        for a in range(4):
            siguiente = vecino(actual, a)
            if valido(mapa, siguiente) and siguiente not in cerrados:
                contador += 1
                f = g + 1 + h(siguiente)
                heapq.heappush(abiertos, (f, h(siguiente), contador, siguiente, g + 1,
                                          a if primero is None else primero))
    raise ValueError("meta inalcanzable")


def nodos_bfs(mapa, origen, meta):
    vistos, cola, i = {origen}, [origen], 0
    while cola[i] != meta:
        for a in range(4):
            siguiente = vecino(cola[i], a)
            if valido(mapa, siguiente) and siguiente not in vistos:
                vistos.add(siguiente)
                cola.append(siguiente)
        i += 1
    return i + 1


def nodos_dfs(mapa, desde, hasta):
    vistos, pila, n = set(), [desde], 0
    while pila:
        actual = pila.pop()
        if actual in vistos:
            continue
        vistos.add(actual)
        n += 1
        if actual == hasta:
            return n
        for a in reversed(range(4)):
            siguiente = vecino(actual, a)
            if valido(mapa, siguiente) and siguiente not in vistos:
                pila.append(siguiente)
    raise ValueError("meta inalcanzable")


def etiquetas_profesores():
    """Mismas parejas y mismo orden que generar_datos, con cada profesor."""
    filas, largos, nodos = [], {"BFS": [], "A*": [], "DFS": []}, {"BFS": [], "A*": [], "DFS": []}
    for indice, mapa in enumerate(MAPAS):
        celdas = libres(mapa)
        for meta in celdas:
            dist = distancias_bfs(mapa, meta)
            padre = arbol_dfs(mapa, meta)
            for pos in celdas:
                if pos == meta:
                    continue
                primero, largo_a, n_a = a_estrella(mapa, pos, meta)
                assert largo_a == dist[pos], "A* debe dar el camino más corto"
                largo_d, celda = 0, pos
                while celda != meta:
                    celda = padre[celda]
                    largo_d += 1
                largos["BFS"].append(dist[pos])
                largos["A*"].append(largo_a)
                largos["DFS"].append(largo_d)
                nodos["BFS"].append(nodos_bfs(mapa, pos, meta))
                nodos["A*"].append(n_a)
                nodos["DFS"].append(nodos_dfs(mapa, meta, pos))
                filas.append((indice, pos, meta, {
                    "BFS": accion_experta(mapa, pos, dist),
                    "A*": primero,
                    "DFS": accion_hacia(pos, padre[pos]),
                }))
    return filas, largos, nodos


def datos_de(filas, profesor, semilla=42, fraccion_prueba=0.2):
    muestras = [(i, p, m, acciones[profesor]) for i, p, m, acciones in filas]
    rng = np.random.default_rng(semilla)
    rng.shuffle(muestras)
    corte = int(len(muestras) * (1 - fraccion_prueba))

    def arrays(parte):
        return (np.stack([codificar(i, p, m) for i, p, m, _ in parte]),
                np.asarray([a for *_, a in parte], dtype=np.int64))

    return (arrays(muestras[:corte]), arrays(muestras[corte:])), muestras[corte:]


def experimento_7():
    print("## Experimento 7: profesores DFS, BFS y A*\n")
    filas, largos, nodos = etiquetas_profesores()
    bfs = np.array([f[3]["BFS"] for f in filas])
    (ref, _), _ = (generar_datos()[:2], None)
    (Xr, yr), _ = datos_de(filas, "BFS")[0]
    assert np.array_equal(Xr, ref[0]) and np.array_equal(yr, ref[1]), "BFS debe igualar generar_datos"

    por_clave = {(f[0], f[1], f[2]): f[3] for f in filas}
    print("Los profesores por sí solos (todos los pares):\n")
    resumen = []
    for prof in ["DFS", "BFS", "A*"]:
        acciones = np.array([f[3][prof] for f in filas])

        def decidir(mundo, prof=prof):
            return por_clave[(mundo.indice, mundo.posicion, mundo.meta)][prof]

        e = evaluar(decidir, [(f[0], f[1], f[2]) for f in filas])
        resumen.append((prof, f"{np.mean(largos[prof]):.2f}", int(np.max(largos[prof])),
                        f"{np.mean(np.array(largos[prof]) / np.array(largos['BFS'])):.2f}",
                        f"{np.mean(nodos[prof]):.1f}", f"{np.mean(acciones == bfs):.1%}",
                        f"{e['exito']:.1%}"))
    tabla(("profesor", "largo medio", "largo máximo", "largo / óptimo", "nodos expandidos",
           "coincide con BFS", "éxito con límite de 45"), resumen)

    print("Neurito entrenado con cada profesor (media de 5 semillas):\n")
    salida = []
    for prof in ["DFS", "BFS", "A*"]:
        datos, casos = datos_de(filas, prof)
        for ep in [400, 1500]:
            rs = resumen_redes(datos, casos, epocas=ep)
            salida.append((prof, ep, f"{media(rs, 'perdida'):.4f}", col(rs, "ent"), col(rs, "pru"),
                           col(rs, "pura", "exito"),
                           f"{media(rs, 'pura', 'pared'):.0f}/{media(rs, 'pura', 'ciclo'):.0f}",
                           f"{media(rs, 'pura', 'pasos'):.1f}", f"{media(rs, 'pura', 'razon'):.2f}",
                           col(rs, "asistida", "exito"), f"{media(rs, 'asistida', 'razon'):.2f}"))
    tabla(("profesor", "épocas", "pérdida", "exactitud entren.", "exactitud prueba", "éxito pura",
           "fallos pura pared/ciclo", "pasos pura", "pura / óptimo", "éxito asistida",
           "asistida / óptimo"), salida)


if __name__ == "__main__":
    ent_prueba = generar_datos()
    datos_guia = ent_prueba[:2]
    casos_prueba = ent_prueba[2]
    # generar_datos solo devuelve los casos de prueba: se reconstruyen los de entrenamiento.
    prueba_set = {(c[0], c[1], c[2]) for c in casos_prueba}
    casos_ent = []
    for ind, mp in enumerate(MAPAS):
        for mt in libres(mp):
            d_bfs = distancias_bfs(mp, mt)
            for ps in libres(mp):
                if ps != mt and (ind, ps, mt) not in prueba_set:
                    casos_ent.append((ind, ps, mt, accion_experta(mp, ps, d_bfs)))
    print(f"Pares totales: {len(casos_ent) + len(casos_prueba)}; "
          f"entrenamiento {len(casos_ent)}; prueba {len(casos_prueba)}\n", flush=True)
    experimento_1(datos_guia, casos_prueba)
    experimento_2_3(datos_guia, casos_prueba)
    experimento_4(datos_guia, casos_prueba)
    experimento_5(datos_guia, casos_prueba)
    experimento_6(datos_guia, casos_ent, casos_prueba)
    experimento_7()

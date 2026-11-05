from pathlib import Path

import numpy as np

# =====================================================================
# CONFIGURACIÓN (cambia estos valores para tus experimentos)
# =====================================================================
LANGUAGE = "es"          # "en" para inglés
HIDDEN_SIZE = 6          # neuronas de la capa oculta del modelo principal
EPOCHS = 1200            # épocas de entrenamiento del modelo principal
LEARNING_RATE = 0.1      # tasa de aprendizaje del modelo principal
RUN_EXPERIMENTS = True   # True: imprime los experimentos en la consola
RUN_GAME = True          # True: abre la ventana de Neurito al final

TEXT = {
    "en": {
        "title": "NEURITO - MY ARTIFICIAL BRAIN",
        "hunger": "Hunger", "energy": "Energy", "boredom": "Boredom",
        "eat": "EAT", "sleep": "SLEEP", "play": "PLAY", "neutral": "CALM",
        "accuracy": "Test accuracy against synthetic policy",
        "epoch": "Epoch", "loss": "Loss",
        "controls1": "H/J: hunger +/-    E/D: energy +/-",
        "controls2": "A/S: boredom +/-    R: random",
        "controls3": "N: calm state    ESC: quit",
    },
    "es": {
        "title": "NEURITO - MI CEREBRO ARTIFICIAL",
        "hunger": "Hambre", "energy": "Energía", "boredom": "Aburrimiento",
        "eat": "COMER", "sleep": "DORMIR", "play": "JUGAR", "neutral": "NEUTRAL",
        "accuracy": "Precisión frente a la política sintética",
        "epoch": "Época", "loss": "Pérdida",
        "controls1": "H/J: hambre +/-    E/D: energía +/-",
        "controls2": "A/S: aburrimiento +/-    R: aleatorio",
        "controls3": "N: estado tranquilo    ESC: salir",
    },
}
labels = TEXT[LANGUAGE]
actions = ("eat", "sleep", "play")


# =====================================================================
# DATOS SINTÉTICOS
# =====================================================================
def policy(data):
    """Política sintética: comer si hambre es lo mayor, dormir si (1 - energía)
    es lo mayor, jugar si aburrimiento es lo mayor."""
    return np.argmax(
        np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1
    )


np.random.seed(42)
X = np.random.rand(3000, 3)
targets = policy(X)
test = np.random.rand(1000, 3)
test_targets = policy(test)


# =====================================================================
# RED NEURONAL
# =====================================================================
def relu(values):
    return np.maximum(0, values)


def softmax(values):
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def predict(data, params):
    W1, b1, W2, b2 = params
    hidden = relu(data @ W1 + b1)
    return softmax(hidden @ W2 + b2)


def train(X, targets, hidden_size=6, epochs=1200, lr=0.1, n_outputs=3,
          seed=0, verbose=False):
    """Entrena con descenso de gradiente (lote completo).
    Devuelve los parámetros y el historial de pérdida."""
    rng = np.random.RandomState(seed)
    n_inputs = X.shape[1]
    W1 = rng.randn(n_inputs, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))
    W2 = rng.randn(hidden_size, n_outputs) * 0.5
    b2 = np.zeros((1, n_outputs))
    history = []
    count = len(X)
    for epoch in range(epochs):
        z1 = X @ W1 + b1
        hidden = relu(z1)
        probabilities = softmax(hidden @ W2 + b2)
        loss = -np.mean(np.log(probabilities[np.arange(count), targets] + 1e-10))
        history.append(loss)
        delta2 = probabilities.copy()
        delta2[np.arange(count), targets] -= 1
        delta2 /= count
        grad_W2 = hidden.T @ delta2
        grad_b2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ W2.T) * (z1 > 0)
        grad_W1 = X.T @ delta1
        grad_b1 = np.sum(delta1, axis=0, keepdims=True)
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        if verbose and epoch % 200 == 0:
            print(f"{labels['epoch']} {epoch} | {labels['loss']}: {loss:.4f}")
    return (W1, b1, W2, b2), history


def accuracy_of(params, data, data_targets):
    return np.mean(np.argmax(predict(data, params), axis=1) == data_targets)


def fmt_probs(p):
    return "  ".join(f"{labels[a]}={p[i] * 100:5.1f}%" for i, a in enumerate(actions))


def line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =====================================================================
# MODELO PRINCIPAL (el que usa el juego)
# =====================================================================
line(f"MODELO PRINCIPAL: {HIDDEN_SIZE} neuronas | {EPOCHS} épocas | lr={LEARNING_RATE}")
params, main_history = train(X, targets, HIDDEN_SIZE, EPOCHS, LEARNING_RATE,
                             seed=42, verbose=True)
print(f"{labels['loss']} final: {main_history[-1]:.4f}")
print(f"{labels['accuracy']}: {accuracy_of(params, test, test_targets):.2%}")


if RUN_EXPERIMENTS:
    # -----------------------------------------------------------------
    # EXPERIMENTO 1: dos necesidades fijas, la tercera varía
    # -----------------------------------------------------------------
    line("EXPERIMENTO 1: dos necesidades fijas y la tercera varía de 0 a 1")
    names = ("hunger", "energy", "boredom")
    base = [0.5, 0.5, 0.5]
    for varied in range(3):
        fixed = [f"{labels[names[i]]}={base[i]}" for i in range(3) if i != varied]
        print(f"\n> Varía {labels[names[varied]]} (fijas: {', '.join(fixed)})")
        values = np.round(np.linspace(0, 1, 21), 2)
        previous = None
        for v in values:
            needs = base.copy()
            needs[varied] = v
            p = predict(np.array([needs]), params)[0]
            act = actions[int(np.argmax(p))]
            mark = "  <-- CAMBIA LA ACCIÓN" if previous and act != previous else ""
            print(f"  {labels[names[varied]]}={v:4.2f} -> {labels[act]:7s} | {fmt_probs(p)}{mark}")
            previous = act
    print("\nSegún la política, la acción cambia cuando la necesidad que varía supera"
          " a las otras: hambre > 0.5, (1 - energía) > 0.5 (energía < 0.5),"
          " aburrimiento > 0.5. Compara ese umbral con el punto donde cambia la red.")

    # -----------------------------------------------------------------
    # EXPERIMENTO 2: prioridades casi iguales
    # -----------------------------------------------------------------
    line("EXPERIMENTO 2: prioridades casi iguales vs. prioridades claras (Softmax)")
    cases = [
        ("Clara: mucha hambre", [0.95, 0.90, 0.10]),
        ("Clara: mucho cansancio", [0.10, 0.05, 0.10]),
        ("Clara: mucho aburrimiento", [0.10, 0.90, 0.95]),
        ("Casi iguales: hambre vs. aburrimiento", [0.60, 0.90, 0.59]),
        ("Casi iguales: hambre vs. cansancio", [0.50, 0.51, 0.10]),
        ("Casi iguales: las tres", [0.50, 0.50, 0.50]),
        ("Casi iguales: las tres (altas)", [0.80, 0.21, 0.79]),
    ]
    for name, needs in cases:
        p = predict(np.array([needs]), params)[0]
        expected = labels[actions[policy(np.array([needs]))[0]]]
        print(f"  {name:40s} {needs} -> {fmt_probs(p)} | máx={p.max() * 100:5.1f}% | política: {expected}")
    print("\nCon prioridades claras, Softmax da una probabilidad alta a una sola acción."
          " Con prioridades casi iguales, las probabilidades se reparten y la red"
          " puede equivocarse respecto a la política.")

    # -----------------------------------------------------------------
    # EXPERIMENTO 3: tamaño de la capa oculta
    # -----------------------------------------------------------------
    line("EXPERIMENTO 3: neuronas en la capa oculta (1200 épocas, lr=0.1)")
    print(f"  {'Neuronas':>8} | {'Pérdida final':>13} | {'Precisión':>9}")
    for h in (3, 6, 10, 20):
        p_h, hist = train(X, targets, hidden_size=h, epochs=1200, lr=0.1, seed=42)
        print(f"  {h:>8} | {hist[-1]:>13.4f} | {accuracy_of(p_h, test, test_targets):>9.2%}")

    # -----------------------------------------------------------------
    # EXPERIMENTO 4: épocas y tasa de aprendizaje
    # -----------------------------------------------------------------
    line("EXPERIMENTO 4: épocas y tasa de aprendizaje (6 neuronas)")

    def diagnose(hist):
        h = np.array(hist)
        if not np.isfinite(h[-1]) or h[-1] > h[0]:
            return "INESTABLE (diverge)"
        tail = h[-100:]
        jumps = np.sum(np.diff(tail) > 0)
        if jumps > 30:
            return "INESTABLE (oscila)"
        improvement = h[-200] - h[-1] if len(h) > 200 else h[0] - h[-1]
        if improvement < 0.005:
            return "SE ESTANCA"
        return "MEJORA"

    print(f"  {'lr':>6} | {'Épocas':>6} | {'Pérdida inicial':>15} | {'Pérdida final':>13} | {'Precisión':>9} | Diagnóstico")
    for lr in (0.01, 0.1, 0.5, 1.0, 5.0):
        for ep in (300, 1200, 3000):
            with np.errstate(all="ignore"):
                p_lr, hist = train(X, targets, hidden_size=6, epochs=ep, lr=lr, seed=42)
                acc = accuracy_of(p_lr, test, test_targets)
            print(f"  {lr:>6} | {ep:>6} | {hist[0]:>15.4f} | {hist[-1]:>13.4f} | {acc:>9.2%} | {diagnose(hist)}")

    # -----------------------------------------------------------------
    # EXPERIMENTO 5: una cuarta acción
    # -----------------------------------------------------------------
    line("EXPERIMENTO 5: ¿qué necesita una cuarta acción? (ejemplo: BAÑARSE)")
    print("  Para agregar una 4.ª acción (BAÑARSE) se necesita:")
    print("   1. Un 4.º dato de entrada: la necesidad 'suciedad' (0 a 1).")
    print("      X pasa de forma (3000, 3) a (3000, 4) y W1 de (3, n) a (4, n).")
    print("   2. Una 4.ª etiqueta: la clase 3 = BAÑARSE, definida en la política:")
    print("      argmax(hambre, 1 - energía, aburrimiento, suciedad).")
    print("   3. Una 4.ª salida en la red: W2 de (n, 3) a (n, 4) y b2 de (1, 3) a (1, 4).")
    print("   4. Ejemplos suficientes y equilibrados de cada una de las 4 clases.")
    print("   5. En el juego: textos, teclas para la nueva barra e imágenes de la acción.")
    X4 = np.random.rand(4000, 4)
    t4 = np.argmax(np.column_stack((X4[:, 0], 1 - X4[:, 1], X4[:, 2], X4[:, 3])), axis=1)
    test4 = np.random.rand(1000, 4)
    test4_t = np.argmax(np.column_stack((test4[:, 0], 1 - test4[:, 1], test4[:, 2], test4[:, 3])), axis=1)
    p4, hist4 = train(X4, t4, hidden_size=10, epochs=1200, lr=0.1, n_outputs=4, seed=42)
    counts = np.bincount(t4, minlength=4)
    print(f"\n  Ejemplos por clase (comer, dormir, jugar, bañarse): {counts.tolist()}")
    print(f"  Red de 4 acciones -> pérdida final: {hist4[-1]:.4f} | "
          f"precisión: {accuracy_of(p4, test4, test4_t):.2%}")
    print("\n" + "=" * 72 + "\nFin de los experimentos.\n")


# =====================================================================
# JUEGO (igual que el original, usando el modelo principal)
# =====================================================================
if RUN_GAME:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("Neurito")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 23)
    asset_dir = Path(__file__).resolve().parent / "assets"
    pet_size = (330, 330)

    def load_frames(names):
        return [
            pygame.transform.smoothscale(
                pygame.image.load(str(asset_dir / name)).convert_alpha(), pet_size
            )
            for name in names
        ]

    frames = {
        "neutral": load_frames(["neutral_1.png", "neutral_2.png", "neutral_3.png"]),
        "eat": load_frames(["comer_1.png", "comer_2.png"]),
        "sleep": load_frames(["dormir_1.png", "dormir_2.png"]),
        "play": load_frames(["jugar_1.png", "jugar_2.png"]),
    }
    needs = [0.2, 0.9, 0.2]
    current_state = "neutral"
    frame_index = 0
    last_frame_time = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    needs[0] = min(1, needs[0] + 0.1)
                elif event.key == pygame.K_j:
                    needs[0] = max(0, needs[0] - 0.1)
                elif event.key == pygame.K_e:
                    needs[1] = min(1, needs[1] + 0.1)
                elif event.key == pygame.K_d:
                    needs[1] = max(0, needs[1] - 0.1)
                elif event.key == pygame.K_a:
                    needs[2] = min(1, needs[2] + 0.1)
                elif event.key == pygame.K_s:
                    needs[2] = max(0, needs[2] - 0.1)
                elif event.key == pygame.K_r:
                    needs = list(np.random.rand(3))
                elif event.key == pygame.K_n:
                    needs = [0.2, 0.9, 0.2]

        probabilities = predict(np.array([needs]), params)[0]
        action = actions[int(np.argmax(probabilities))]
        if needs[0] < 0.4 and needs[1] > 0.6 and needs[2] < 0.4:
            state = "neutral"  # Regla de visualización; la red tiene tres salidas.
        else:
            state = action

        now = pygame.time.get_ticks()
        if state != current_state:
            current_state, frame_index, last_frame_time = state, 0, now
        elif now - last_frame_time >= 500:
            frame_index = (frame_index + 1) % len(frames[state])
            last_frame_time = now

        screen.fill((250, 245, 252))
        title = font.render(labels["title"], True, (55, 45, 75))
        screen.blit(title, (45, 28))

        colors = [(235, 125, 75), (70, 175, 115), (105, 125, 225)]
        for index, key in enumerate(("hunger", "energy", "boredom")):
            value = needs[index]
            y = 125 + index * 95
            text = small_font.render(
                f"{labels[key]}: {value * 100:.0f}%", True, (35, 40, 55)
            )
            screen.blit(text, (50, y - 27))
            pygame.draw.rect(screen, (215, 220, 230), (50, y, 290, 22), border_radius=7)
            pygame.draw.rect(
                screen, colors[index], (50, y, int(290 * value), 22), border_radius=7
            )

        pet = frames[state][frame_index]
        screen.blit(pet, pet.get_rect(center=(650, 235)))
        screen.blit(font.render(labels[state], True, (65, 50, 85)), (585, 420))
        for index, action_name in enumerate(actions):
            score = small_font.render(
                f"{labels[action_name]}: {probabilities[index] * 100:.1f}%",
                True, (55, 50, 75)
            )
            screen.blit(score, (50, 390 + index * 28))

        screen.blit(small_font.render(labels["controls1"], True, (70, 70, 90)), (50, 495))
        screen.blit(small_font.render(labels["controls2"], True, (70, 70, 90)), (50, 525))
        screen.blit(small_font.render(labels["controls3"], True, (70, 70, 90)), (50, 555))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
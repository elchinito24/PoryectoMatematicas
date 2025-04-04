import numpy as np
import matplotlib.pyplot as plt

# Función objetivo
def f(x):
    return x**4 - 3*x**3 + 2

# Derivada (gradiente)
def df(x):
    return 4*x**3 - 9*x**2

# Segunda derivada (Hessiana en 1D)
def ddf(x):
    return 12*x**2 - 18*x

# Gradiente Descendente
def gradiente_descendente(x0, lr=0.01, max_iter=50):
    x_values = [x0]
    for _ in range(max_iter):
        grad = df(x_values[-1])
        x_new = x_values[-1] - lr * grad
        x_values.append(x_new)
        if abs(grad) < 1e-6:
            break
    return x_values

# Newton-Raphson
def newton_raphson(x0, max_iter=50):
    x_values = [x0]
    for _ in range(max_iter):
        grad = df(x_values[-1])
        hess = ddf(x_values[-1])
        if abs(hess) < 1e-6:
            break
        x_new = x_values[-1] - grad / hess
        x_values.append(x_new)
        if abs(grad) < 1e-6:
            break
    return x_values

# --- Parámetros del usuario ---
x0 = float(input("Ingrese el valor inicial x0: "))
learning_rate = float(input("Ingrese la tasa de aprendizaje (ej. 0.01): "))
iteraciones = int(input("Ingrese el número máximo de iteraciones: "))

# Ejecutar métodos
gd_path = gradiente_descendente(x0, lr=learning_rate, max_iter=iteraciones)
nr_path = newton_raphson(x0, max_iter=iteraciones)

# Generar datos para la curva de la función
x_vals = np.linspace(min(min(gd_path), min(nr_path)) - 1, max(max(gd_path), max(nr_path)) + 1, 400)
y_vals = f(x_vals)

# --- Gráfico ---
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x)', color='gray')

# Trayectorias
plt.plot(gd_path, [f(x) for x in gd_path], 'o-', label='Gradiente Descendente', color='blue')
plt.plot(nr_path, [f(x) for x in nr_path], 'o-', label='Newton-Raphson', color='green')

# Detalles
plt.title("Comparación: Gradiente Descendente vs Newton-Raphson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

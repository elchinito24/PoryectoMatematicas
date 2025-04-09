import numpy as np
import tkinter as tk
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ajuste de fuente para Matplotlib
plt.rcParams.update({"font.size": 10})


def gradiente_descendente_2d(f_grad, x0, lr, iterations):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(iterations):
        grad = f_grad(x)
        x -= lr * grad
        history.append(x.copy())
    return np.array(history)


def newton_raphson_2d(f_grad, f_hess, x0, iterations):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(iterations):
        grad = f_grad(x)
        hess = f_hess(x)
        # Verificar si la Hessiana es invertible
        if abs(np.linalg.det(hess)) < 1e-8:
            break
        # Resolver H * delta = grad
        delta = np.linalg.solve(hess, grad)
        x -= delta
        history.append(x.copy())
    return np.array(history)


class AppComparativaOptimizacion2D:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparativa 2D: Gradiente Descendente vs Newton-Raphson")
        self.crear_widgets()
        self.actualizar_graficos()

    def crear_widgets(self):
        # Fuente grande para labels, inputs y botones
        font_config = ("Arial", 36)

        frame_param = tk.Frame(self.root, padx=10, pady=10)
        frame_param.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Función
        tk.Label(frame_param, text="Función f(x,y):", font=font_config).grid(
            row=0, column=0, sticky="e"
        )
        self.func_str = tk.StringVar(value="x**2 + y**2 + x*y - 4*x - 2*y + 4")
        tk.Entry(
            frame_param, textvariable=self.func_str, font=font_config, width=20
        ).grid(row=0, column=1, columnspan=3)

        # Punto inicial x0, y0
        tk.Label(frame_param, text="x0:", font=font_config).grid(
            row=1, column=0, sticky="e"
        )
        self.x0 = tk.DoubleVar(value=1.0)
        tk.Entry(frame_param, textvariable=self.x0, font=font_config, width=8).grid(
            row=1, column=1
        )
        tk.Label(frame_param, text="y0:", font=font_config).grid(
            row=1, column=2, sticky="e"
        )
        self.y0 = tk.DoubleVar(value=1.0)
        tk.Entry(frame_param, textvariable=self.y0, font=font_config, width=8).grid(
            row=1, column=3
        )

        # Learning Rate y iteraciones
        tk.Label(frame_param, text="Taza de Aprendizaje (GD):", font=font_config).grid(
            row=2, column=0, sticky="e"
        )
        self.lr = tk.DoubleVar(value=0.1)
        tk.Entry(frame_param, textvariable=self.lr, font=font_config, width=8).grid(
            row=2, column=1
        )
        tk.Label(frame_param, text="Iter GD:", font=font_config).grid(
            row=2, column=2, sticky="e"
        )
        self.iter_gd = tk.IntVar(value=50)
        tk.Entry(
            frame_param, textvariable=self.iter_gd, font=font_config, width=8
        ).grid(row=2, column=3)
        tk.Label(frame_param, text="Iter NR:", font=font_config).grid(
            row=2, column=4, sticky="e"
        )
        self.iter_nr = tk.IntVar(value=10)
        tk.Entry(
            frame_param, textvariable=self.iter_nr, font=font_config, width=8
        ).grid(row=2, column=5)

        # Botón actualizar
        tk.Button(
            frame_param,
            text="Actualizar",
            font=font_config,
            command=self.actualizar_graficos,
        ).grid(row=1, column=5, padx=10)

        # Gráficos
        frame_gd = tk.LabelFrame(
            self.root, text="Gradiente Descendente", padx=5, pady=5
        )
        frame_gd.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.fig_gd, self.ax_gd = plt.subplots()
        self.canvas_gd = FigureCanvasTkAgg(self.fig_gd, master=frame_gd)
        self.canvas_gd.get_tk_widget().pack(fill="both", expand=True)

        frame_nr = tk.LabelFrame(self.root, text="Newton-Raphson", padx=5, pady=5)
        frame_nr.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.fig_nr, self.ax_nr = plt.subplots()
        self.canvas_nr = FigureCanvasTkAgg(self.fig_nr, master=frame_nr)
        self.canvas_nr.get_tk_widget().pack(fill="both", expand=True)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

    def actualizar_graficos(self):
        func_str = self.func_str.get()
        x0 = self.x0.get()
        y0 = self.y0.get()
        lr = self.lr.get()
        iter_gd = self.iter_gd.get()
        iter_nr = self.iter_nr.get()

        # Definir símbolos y expresiones
        x, y = sp.symbols("x y")
        f_expr = sp.sympify(func_str)
        grad_expr = [sp.diff(f_expr, var) for var in (x, y)]
        hess_expr = [[sp.diff(g, var) for var in (x, y)] for g in grad_expr]

        # Lambdify
        f_lambda = sp.lambdify((x, y), f_expr, "numpy")
        grad_lambda = sp.lambdify((x, y), grad_expr, "numpy")
        hess_lambda = sp.lambdify((x, y), hess_expr, "numpy")

        # Ejecutar métodos
        history_gd = gradiente_descendente_2d(
            lambda p: np.array(grad_lambda(p[0], p[1])), [x0, y0], lr, iter_gd
        )
        history_nr = newton_raphson_2d(
            lambda p: np.array(grad_lambda(p[0], p[1])),
            lambda p: np.array(hess_lambda(p[0], p[1])),
            [x0, y0],
            iter_nr,
        )

        # Plot
        self.plot_comparison_2d(f_lambda, history_gd, history_nr)

    def plot_comparison_2d(self, f_lambda, history_gd, history_nr):
        # Rango dinámico basado en historial
        xs = np.linspace(
            min(history_gd[:, 0].min(), history_nr[:, 0].min()) - 1,
            max(history_gd[:, 0].max(), history_nr[:, 0].max()) + 1,
            200,
        )
        ys = np.linspace(
            min(history_gd[:, 1].min(), history_nr[:, 1].min()) - 1,
            max(history_gd[:, 1].max(), history_nr[:, 1].max()) + 1,
            200,
        )
        X, Y = np.meshgrid(xs, ys)
        Z = f_lambda(X, Y)

        # Contour + trayectoria GD
        self.ax_gd.clear()
        cs = self.ax_gd.contour(X, Y, Z, levels=30)
        self.ax_gd.clabel(cs, inline=True, fontsize=18)
        self.ax_gd.plot(history_gd[:, 0], history_gd[:, 1], marker="o", label="GD")
        self.ax_gd.set_title("Gradiente Descendente", {"fontsize": 30})
        self.ax_gd.set_xlabel("x")
        self.ax_gd.set_ylabel("y")
        self.ax_gd.legend()
        self.ax_gd.grid(True)

        # Contour + trayectoria NR
        self.ax_nr.clear()
        cs2 = self.ax_nr.contour(X, Y, Z, levels=30)
        self.ax_nr.clabel(cs2, inline=True, fontsize=18)
        self.ax_nr.plot(history_nr[:, 0], history_nr[:, 1], marker="o", label="NR")
        self.ax_nr.set_title("Newton-Raphson", {"fontsize": 30})
        self.ax_nr.set_xlabel("x")
        self.ax_nr.set_ylabel("y")
        self.ax_nr.legend()
        self.ax_nr.grid(True)

        self.canvas_gd.draw()
        self.canvas_nr.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = AppComparativaOptimizacion2D(root)
    root.mainloop()

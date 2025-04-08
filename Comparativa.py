import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp

plt.rcParams.update({"font.size": 14})  # Tamaño para matplotlib

def verificar_hessiana(hess_func, x0):
    hess_val = hess_func(x0)
    return np.abs(hess_val) < 1e-8

def gradiente_descendente(f, df, x0, lr=0.001, iteraciones=100):
    x = x0
    historial = [x]
    for _ in range(iteraciones):
        x = x - lr * df(x)
        historial.append(x)
    return np.array(historial)

def newton_raphson(f, df, d2f, x0, iteraciones=10):
    x = x0
    historial = [x]
    for _ in range(iteraciones):
        derivada_segunda = d2f(x)
        if np.abs(derivada_segunda) < 1e-8:
            break
        x = x - df(x) / derivada_segunda
        historial.append(x)
    return np.array(historial)

class AppComparativaOptimizacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparativa: Gradiente Descendente vs Newton-Raphson")
        self.crear_widgets()
        self.actualizar_graficos()

    def crear_widgets(self):
        font_config = ("Arial", 32)

        frame_param = tk.Frame(self.root, padx=10, pady=10)
        frame_param.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Función
        tk.Label(frame_param, text="Función:", font=font_config).grid(row=0, column=0, sticky="e")
        self.func_str = tk.StringVar(value="x**4 - 3*x**3 + 2")
        tk.Entry(frame_param, textvariable=self.func_str, font=font_config, width=18).grid(row=0, column=1)

        # Punto inicial
        tk.Label(frame_param, text="Punto inicial (x0):", font=font_config).grid(row=1, column=0, sticky="e")
        self.x0 = tk.DoubleVar(value=3.0)
        tk.Entry(frame_param, textvariable=self.x0, font=font_config, width=6).grid(row=1, column=1)

        # Learning Rate
        tk.Label(frame_param, text="Learning Rate (GD):", font=font_config).grid(row=1, column=2, sticky="e")
        self.lr = tk.DoubleVar(value=0.001)
        tk.Entry(frame_param, textvariable=self.lr, font=font_config, width=6).grid(row=1, column=3)

        # Iteraciones GD
        tk.Label(frame_param, text="Iteraciones GD:", font=font_config).grid(row=1, column=4, sticky="e")
        self.iter_gd = tk.IntVar(value=100)
        tk.Entry(frame_param, textvariable=self.iter_gd, font=font_config, width=6).grid(row=1, column=5)

        # Iteraciones NR
        tk.Label(frame_param, text="Iteraciones NR:", font=font_config).grid(row=1, column=6, sticky="e")
        self.iter_nr = tk.IntVar(value=10)
        tk.Entry(frame_param, textvariable=self.iter_nr, font=font_config, width=6).grid(row=1, column=7)

        # Botón actualizar
        tk.Button(frame_param, text="Actualizar", font=font_config, command=self.actualizar_graficos).grid(row=1, column=8, padx=10)

        # Gráfico GD
        frame_gd = tk.LabelFrame(self.root, text="Método Gradiente Descendente", padx=10, pady=10, font=font_config)
        frame_gd.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.fig_gd, self.ax_gd = plt.subplots(figsize=(6, 4))
        self.canvas_gd = FigureCanvasTkAgg(self.fig_gd, master=frame_gd)
        self.canvas_gd.get_tk_widget().pack(fill="both", expand=True)

        # Gráfico NR
        frame_nr = tk.LabelFrame(self.root, text="Método Newton-Raphson", padx=10, pady=10, font=font_config)
        frame_nr.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        self.fig_nr, self.ax_nr = plt.subplots(figsize=(6, 4))
        self.canvas_nr = FigureCanvasTkAgg(self.fig_nr, master=frame_nr)
        self.canvas_nr.get_tk_widget().pack(fill="both", expand=True)

        # Texto explicativo
        frame_info = tk.Frame(self.root, padx=10, pady=20)
        frame_info.grid(row=3, column=0, columnspan=2)
        self.info_label = tk.Label(frame_info, text="", font=font_config, wraplength=1800, justify="center")
        self.info_label.pack()

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

    def actualizar_graficos(self):
        func_str = self.func_str.get()
        x0 = self.x0.get()
        lr = self.lr.get()
        iter_gd = self.iter_gd.get()
        iter_nr = self.iter_nr.get()

        x = sp.symbols('x')
        f_expr = sp.sympify(func_str)
        df_expr = sp.diff(f_expr, x)
        d2f_expr = sp.diff(df_expr, x)

        f_lambda = sp.lambdify(x, f_expr, "numpy")
        df_lambda = sp.lambdify(x, df_expr, "numpy")
        d2f_lambda = sp.lambdify(x, d2f_expr, "numpy")

        if verificar_hessiana(d2f_lambda, x0):
            self.info_label.config(
                text="La Hessiana es cero en el punto inicial. Se recomienda usar Gradiente Descendente."
            )
            historial_gd = gradiente_descendente(f_lambda, df_lambda, x0, lr, iter_gd)
            self.plot_comparison(f_lambda, historial_gd, None)
        else:
            self.info_label.config(
                text="La Hessiana no es cero. Newton-Raphson puede converger más rápido."
            )
            historial_gd = gradiente_descendente(f_lambda, df_lambda, x0, lr, iter_gd)
            historial_nr = newton_raphson(f_lambda, df_lambda, d2f_lambda, x0, iter_nr)
            self.plot_comparison(f_lambda, historial_gd, historial_nr)

    def plot_comparison(self, f_lambda, historial_gd, historial_nr):
        x_vals = np.linspace(-1, 4, 400)
        y_vals = f_lambda(x_vals)

        self.ax_gd.clear()
        self.ax_gd.plot(x_vals, y_vals, "k-", label="f(x)")
        self.ax_gd.scatter(historial_gd, f_lambda(historial_gd), c="blue", label="GD", alpha=0.7)
        self.ax_gd.set_title("Gradiente Descendente")
        self.ax_gd.set_xlabel("x")
        self.ax_gd.set_ylabel("f(x)")
        self.ax_gd.legend()
        self.ax_gd.grid(True)
        self.canvas_gd.draw()

        if historial_nr is not None:
            self.ax_nr.clear()
            self.ax_nr.plot(x_vals, y_vals, "k-", label="f(x)")
            self.ax_nr.scatter(historial_nr, f_lambda(historial_nr), c="red", label="NR", alpha=0.7)
            self.ax_nr.set_title("Newton-Raphson")
            self.ax_nr.set_xlabel("x")
            self.ax_nr.set_ylabel("f(x)")
            self.ax_nr.legend()
            self.ax_nr.grid(True)
            self.canvas_nr.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = AppComparativaOptimizacion(root)
    root.mainloop()

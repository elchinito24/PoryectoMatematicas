# import numpy as np
# import tkinter as tk
# from tkinter import ttk
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import sympy as sp

# # Actualizar tamaño de fuente en Matplotlib
# plt.rcParams.update({"font.size": 10})


# # Algoritmo de Gradiente Descendente (usa la primera derivada)
# def gradiente_descendente(f, df, x0, lr=0.001, iteraciones=100):
#     x = x0
#     historial = [x]
#     for _ in range(iteraciones):
#         x = x - lr * df(x)
#         historial.append(x)
#     return np.array(historial)


# # Algoritmo Newton-Raphson (usa la segunda derivada)
# def newton_raphson(f, df, d2f, x0, iteraciones=10):
#     x = x0
#     historial = [x]
#     for _ in range(iteraciones):
#         # Evitar división por cero en caso de que d2f(x) se acerque a 0
#         derivada_segunda = d2f(x)
#         if np.abs(derivada_segunda) < 1e-8:
#             break
#         x = x - df(x) / derivada_segunda
#         historial.append(x)
#     return np.array(historial)


# # Clase de la aplicación comparativa
# class AppComparativaOptimizacion:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Comparativa: Gradiente Descendente vs Newton-Raphson")
#         self.crear_widgets()
#         self.actualizar_graficos()

#     def crear_widgets(self):
#         # Frame de parámetros de entrada
#         frame_param = ttk.Frame(self.root, padding=10)
#         frame_param.grid(row=0, column=0, columnspan=2, sticky="ew")

#         ttk.Label(frame_param, text="Función:").grid(
#             row=0, column=0, padx=5, pady=5, sticky="e"
#         )
#         self.func_str = tk.StringVar(value="x**4 - 3*x**3 + 2")
#         ttk.Entry(frame_param, textvariable=self.func_str, width=30).grid(
#             row=0, column=1, padx=5, pady=5
#         )

#         ttk.Label(frame_param, text="Punto inicial (x0):").grid(
#             row=1, column=0, padx=5, pady=5, sticky="e"
#         )
#         self.x0 = tk.DoubleVar(value=3.0)
#         ttk.Entry(frame_param, textvariable=self.x0, width=10).grid(
#             row=1, column=1, padx=5, pady=5
#         )

#         ttk.Label(frame_param, text="Learning Rate (GD):").grid(
#             row=1, column=2, padx=5, pady=5, sticky="e"
#         )
#         self.lr = tk.DoubleVar(value=0.001)
#         ttk.Entry(frame_param, textvariable=self.lr, width=10).grid(
#             row=1, column=3, padx=5, pady=5
#         )

#         ttk.Label(frame_param, text="Iteraciones Gradiente Descendente:").grid(
#             row=1, column=4, padx=5, pady=5, sticky="e"
#         )
#         self.iter_gd = tk.IntVar(value=100)
#         ttk.Entry(frame_param, textvariable=self.iter_gd, width=10).grid(
#             row=1, column=5, padx=5, pady=5
#         )

#         ttk.Label(frame_param, text="Iteraciones Newton-Raphson:").grid(
#             row=1, column=6, padx=5, pady=5, sticky="e"
#         )
#         self.iter_nr = tk.IntVar(value=10)
#         ttk.Entry(frame_param, textvariable=self.iter_nr, width=10).grid(
#             row=1, column=7, padx=5, pady=5
#         )

#         # Botón para actualizar los gráficos
#         ttk.Button(
#             frame_param, text="Actualizar Gráficos", command=self.actualizar_graficos
#         ).grid(row=1, column=8, padx=10, pady=5)

#         # Frame para el gráfico del Gradiente Descendente
#         frame_gd = ttk.LabelFrame(
#             self.root, text="Método Gradiente Descendente", padding=10
#         )
#         frame_gd.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
#         self.fig_gd, self.ax_gd = plt.subplots(figsize=(6, 4))
#         self.canvas_gd = FigureCanvasTkAgg(self.fig_gd, master=frame_gd)
#         self.canvas_gd.get_tk_widget().pack(fill="both", expand=True)

#         # Frame para el gráfico de Newton-Raphson
#         frame_nr = ttk.LabelFrame(self.root, text="Método Newton-Raphson", padding=10)
#         frame_nr.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
#         self.fig_nr, self.ax_nr = plt.subplots(figsize=(6, 4))
#         self.canvas_nr = FigureCanvasTkAgg(self.fig_nr, master=frame_nr)
#         self.canvas_nr.get_tk_widget().pack(fill="both", expand=True)

#         # Frame inferior para explicación comparativa
#         frame_info = ttk.Frame(self.root, padding=10)
#         frame_info.grid(row=3, column=0, columnspan=2, sticky="ew")
#         info_text = (
#             "El método Newton-Raphson es preferido en muchos casos ya que utiliza la segunda derivada para "
#             "ajustar la magnitud y dirección del paso, lo que permite una convergencia más rápida y precisa "
#             "al mínimo, en comparación con el Gradiente Descendente que solo utiliza la primera derivada y "
#             "puede requerir una tasa de aprendizaje muy precisa y más iteraciones para converger."
#         )
#         ttk.Label(frame_info, text=info_text, wraplength=800, justify="center").pack()

#         # Configurar expansión de grillas
#         self.root.columnconfigure(0, weight=1)
#         self.root.columnconfigure(1, weight=1)
#         self.root.rowconfigure(2, weight=1)

#     def actualizar_graficos(self):
#         # Obtener parámetros desde las entradas
#         func_str = self.func_str.get()
#         x0 = self.x0.get()
#         lr = self.lr.get()
#         iter_gd = self.iter_gd.get()
#         iter_nr = self.iter_nr.get()

#         # Parsear la función ingresada como texto
#         x = sp.symbols('x')
#         f_expr = sp.sympify(func_str)  # Convertir la cadena a expresión simbólica

#         # Derivadas de la función
#         df_expr = sp.diff(f_expr, x)
#         d2f_expr = sp.diff(df_expr, x)

#         # Funciones lambda para evaluar
#         f_lambda = sp.lambdify(x, f_expr, "numpy")
#         df_lambda = sp.lambdify(x, df_expr, "numpy")
#         d2f_lambda = sp.lambdify(x, d2f_expr, "numpy")

#         # Calcular trayectorias de ambos métodos
#         historial_gd = gradiente_descendente(f_lambda, df_lambda, x0, lr, iter_gd)
#         historial_nr = newton_raphson(f_lambda, df_lambda, d2f_lambda, x0, iter_nr)

#         # Dominio para graficar la función
#         x_vals = np.linspace(-1, 4, 400)
#         y_vals = f_lambda(x_vals)

#         # Actualizar gráfico de Gradiente Descendente
#         self.ax_gd.clear()
#         self.ax_gd.plot(x_vals, y_vals, "k-", label="f(x)")
#         self.ax_gd.scatter(
#             historial_gd,
#             f_lambda(historial_gd),
#             c="blue",
#             label=f"GD ({len(historial_gd)-1} iter)",
#             alpha=0.7,
#         )
#         self.ax_gd.set_title("Gradiente Descendente")
#         self.ax_gd.set_xlabel("x")
#         self.ax_gd.set_ylabel("f(x)")
#         self.ax_gd.legend()
#         self.ax_gd.grid(True)
#         self.canvas_gd.draw()

#         # Actualizar gráfico de Newton-Raphson
#         self.ax_nr.clear()
#         self.ax_nr.plot(x_vals, y_vals, "k-", label="f(x)")
#         self.ax_nr.scatter(
#             historial_nr,
#             f_lambda(historial_nr),
#             c="red",
#             label=f"NR ({len(historial_nr)-1} iter)",
#             alpha=0.7,
#         )
#         self.ax_nr.set_title("Newton-Raphson")
#         self.ax_nr.set_xlabel("x")
#         self.ax_nr.set_ylabel("f(x)")
#         self.ax_nr.legend()
#         self.ax_nr.grid(True)
#         self.canvas_nr.draw()


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AppComparativaOptimizacion(root)
#     root.mainloop()



import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp

# Actualizar tamaño de fuente en Matplotlib
plt.rcParams.update({"font.size": 10})

# Función para evaluar si la Hessiana es cero
def verificar_hessiana(hess_func, x0):
    hess_val = hess_func(x0)
    if np.abs(hess_val) < 1e-8:
        return True  # Indica que la Hessiana es cero
    return False

# Algoritmo de Gradiente Descendente (usa la primera derivada)
def gradiente_descendente(f, df, x0, lr=0.001, iteraciones=100):
    x = x0
    historial = [x]
    for _ in range(iteraciones):
        x = x - lr * df(x)
        historial.append(x)
    return np.array(historial)

# Algoritmo Newton-Raphson (usa la segunda derivada)
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

# Clase de la aplicación comparativa
class AppComparativaOptimizacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparativa: Gradiente Descendente vs Newton-Raphson")
        self.crear_widgets()
        self.actualizar_graficos()

    def crear_widgets(self):
        # Frame de parámetros de entrada
        font_config = ("Arial", 36)
        frame_param = ttk.Frame(self.root, padding=10)
        frame_param.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(frame_param, text="Función:").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        self.func_str = tk.StringVar(value="x**4 - 3*x**3 + 2")
        ttk.Entry(frame_param, textvariable=self.func_str, width=30).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(frame_param, text="Punto inicial (x0):").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        self.x0 = tk.DoubleVar(value=3.0)
        ttk.Entry(frame_param, textvariable=self.x0, width=10).grid(
            row=1, column=1, padx=5, pady=5
        )

        ttk.Label(frame_param, text="Learning Rate (GD):").grid(
            row=1, column=2, padx=5, pady=5, sticky="e"
        )
        self.lr = tk.DoubleVar(value=0.001)
        ttk.Entry(frame_param, textvariable=self.lr, width=10).grid(
            row=1, column=3, padx=5, pady=5
        )

        ttk.Label(frame_param, text="Iteraciones Gradiente Descendente:").grid(
            row=1, column=4, padx=5, pady=5, sticky="e"
        )
        self.iter_gd = tk.IntVar(value=100)
        ttk.Entry(frame_param, textvariable=self.iter_gd, width=10).grid(
            row=1, column=5, padx=5, pady=5
        )

        ttk.Label(frame_param, text="Iteraciones Newton-Raphson:").grid(
            row=1, column=6, padx=5, pady=5, sticky="e"
        )
        self.iter_nr = tk.IntVar(value=10)
        ttk.Entry(frame_param, textvariable=self.iter_nr, width=10).grid(
            row=1, column=7, padx=5, pady=5
        )

        # Botón para actualizar los gráficos
        ttk.Button(
            frame_param, text="Actualizar Gráficos", command=self.actualizar_graficos
        ).grid(row=1, column=8, padx=10, pady=5)

        # Frame para el gráfico del Gradiente Descendente
        frame_gd = ttk.LabelFrame(
            self.root, text="Método Gradiente Descendente", padding=10
        )
        frame_gd.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.fig_gd, self.ax_gd = plt.subplots(figsize=(6, 4))
        self.canvas_gd = FigureCanvasTkAgg(self.fig_gd, master=frame_gd)
        self.canvas_gd.get_tk_widget().pack(fill="both", expand=True)

        # Frame para el gráfico de Newton-Raphson
        frame_nr = ttk.LabelFrame(self.root, text="Método Newton-Raphson", padding=10)
        frame_nr.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        self.fig_nr, self.ax_nr = plt.subplots(figsize=(6, 4))
        self.canvas_nr = FigureCanvasTkAgg(self.fig_nr, master=frame_nr)
        self.canvas_nr.get_tk_widget().pack(fill="both", expand=True)

        # Frame inferior para explicación comparativa
        frame_info = ttk.Frame(self.root, padding=10)
        frame_info.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.info_label = ttk.Label(frame_info, text="", wraplength=800, justify="center")
        self.info_label.pack()

        # Configurar expansión de grillas
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

    def actualizar_graficos(self):
        # Obtener parámetros desde las entradas
        func_str = self.func_str.get()
        x0 = self.x0.get()
        lr = self.lr.get()
        iter_gd = self.iter_gd.get()
        iter_nr = self.iter_nr.get()

        # Parsear la función ingresada como texto
        x = sp.symbols('x')
        f_expr = sp.sympify(func_str)

        # Derivadas de la función
        df_expr = sp.diff(f_expr, x)
        d2f_expr = sp.diff(df_expr, x)

        # Funciones lambda para evaluar
        f_lambda = sp.lambdify(x, f_expr, "numpy")
        df_lambda = sp.lambdify(x, df_expr, "numpy")
        d2f_lambda = sp.lambdify(x, d2f_expr, "numpy")

        # Verificar si la Hessiana es cero
        hessiana_cero = verificar_hessiana(d2f_lambda, x0)

        # Mostrar el mensaje adecuado
        if hessiana_cero:
            self.info_label.config(
                text="La Hessiana es cero en el punto inicial. El Gradiente Descendente es el método recomendado."
            )
            # Solo ejecutar Gradiente Descendente
            historial_gd = gradiente_descendente(f_lambda, df_lambda, x0, lr, iter_gd)
            self.plot_comparison(f_lambda, historial_gd, None)
        else:
            self.info_label.config(
                text="La Hessiana no es cero. El método de Newton-Raphson es adecuado y puede ofrecer una convergencia más rápida."
            )
            # Ejecutar ambos métodos
            historial_gd = gradiente_descendente(f_lambda, df_lambda, x0, lr, iter_gd)
            historial_nr = newton_raphson(f_lambda, df_lambda, d2f_lambda, x0, iter_nr)
            self.plot_comparison(f_lambda, historial_gd, historial_nr)

    def plot_comparison(self, f_lambda, historial_gd, historial_nr):
        # Dominio para graficar la función
        x_vals = np.linspace(-1, 4, 400)
        y_vals = f_lambda(x_vals)

        # Actualizar gráfico de Gradiente Descendente
        self.ax_gd.clear()
        self.ax_gd.plot(x_vals, y_vals, "k-", label="f(x)")
        self.ax_gd.scatter(
            historial_gd,
            f_lambda(historial_gd),
            c="blue",
            label=f"GD ({len(historial_gd)-1} iter)",
            alpha=0.7,
        )
        self.ax_gd.set_title("Gradiente Descendente")
        self.ax_gd.set_xlabel("x")
        self.ax_gd.set_ylabel("f(x)")
        self.ax_gd.legend()
        self.ax_gd.grid(True)
        self.canvas_gd.draw()

        # Actualizar gráfico de Newton-Raphson
        if historial_nr is not None:
            self.ax_nr.clear()
            self.ax_nr.plot(x_vals, y_vals, "k-", label="f(x)")
            self.ax_nr.scatter(
                historial_nr,
                f_lambda(historial_nr),
                c="red",
                label=f"NR ({len(historial_nr)-1} iter)",
                alpha=0.7,
            )
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

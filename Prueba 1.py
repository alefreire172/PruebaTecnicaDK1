# ======================================================
# PRUEBA TÉCNICA 1 - ESTIMACIÓN DE COSTOS DE EQUIPOS
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear carpeta de resultados
if not os.path.exists("Resultados"):
    os.makedirs("Resultados")
    print("Carpeta 'Resultados' creada")

# ======================================================
# CARGA Y LIMPIEZA DE DATOS
# ======================================================

data_path = "Datos"

def cargar_archivo(nombre):
    ruta = os.path.join(data_path, nombre)
    
    # intenta primero con coma
    df = pd.read_csv(ruta)
    
    # si quedó en una sola columna → probar con ;
    if len(df.columns) == 1:
        df = pd.read_csv(ruta, sep=";")
    
    # limpiar columnas
    df.columns = df.columns.str.strip().str.capitalize()
    
    # si columnas están invertidas
    if df.columns.tolist() == ["Price", "Date"]:
        df = df[["Date", "Price"]]
    
    return df

print("Cargando archivos...")
x = cargar_archivo("X.csv")
y = cargar_archivo("Y.csv")
z = cargar_archivo("Z.csv")

for df_temp in [x, y, z]:
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='mixed', dayfirst=True, errors='coerce')
    df_temp.sort_values('Date', inplace=True)

# convertir precios a numéricos
for df_temp in [x, y, z]:
    df_temp["Price"] = (
        df_temp["Price"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df_temp["Price"] = pd.to_numeric(df_temp["Price"], errors="coerce")

# renombrar precios
x = x.rename(columns={"Price": "X"})
y = y.rename(columns={"Price": "Y"})
z = z.rename(columns={"Price": "Z"})

# unir datasets por fecha
df = x.merge(y, on="Date", how="inner").merge(z, on="Date", how="inner")

print(f"Datos cargados correctamente")
print(f"Rango de fechas: {df['Date'].min()} a {df['Date'].max()}")
print(f"Total registros: {len(df)}")

# ======================================================
# PASO 2 — CÁLCULO DE EQUIPOS
# ======================================================

df["Equipo1"] = 0.2 * df["X"] + 0.8 * df["Y"]
df["Equipo2"] = (df["X"] + df["Y"] + df["Z"]) / 3

print("\nEquipos calculados")

# ======================================================
# PASO 3 — CLASE TIME SERIES PREDICTOR
# ======================================================

class TimeSeriesPredictor:
    """Clase para predicción de series temporales"""
    
    def __init__(self, df, equipo_col):
        self.df = df.copy()
        self.equipo_col = equipo_col
        self.models = {}
        self.results = {}
        
    def prepare_data(self, n_lags=3):
        """Prepara datos con variables de rezago"""
        data = self.df[['Date', self.equipo_col]].copy()
        data['t'] = np.arange(len(data))
        
        # Crear lags
        for i in range(1, n_lags + 1):
            data[f'lag_{i}'] = data[self.equipo_col].shift(i)
        
        # Características temporales
        data['month'] = data['Date'].dt.month
        data['year'] = data['Date'].dt.year
        
        # Dummies para meses
        month_dummies = pd.get_dummies(data['month'], prefix='month')
        data = pd.concat([data, month_dummies], axis=1)
        
        return data.dropna()
    
    def train_test_split(self, data, test_size=0.2):
        """Divide datos respetando orden temporal"""
        split_idx = int(len(data) * (1 - test_size))
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        return train, test, split_idx
    
    def fit_linear_trend(self):
        """Modelo de tendencia lineal"""
        data = self.prepare_data(n_lags=0)
        train, test, _ = self.train_test_split(data)
        
        feature_cols = ['t']
        
        model = LinearRegression()
        model.fit(train[feature_cols], train[self.equipo_col])
        
        self.models['linear_trend'] = {
            'model': model,
            'features': feature_cols,
            'train': train,
            'test': test
        }
        
        return self
    
    def fit_lag_model(self, n_lags=3):
        """Modelo con variables de rezago"""
        data = self.prepare_data(n_lags=n_lags)
        train, test, _ = self.train_test_split(data)
        
        feature_cols = ['t'] + [f'lag_{i}' for i in range(1, n_lags+1)]
        
        model = LinearRegression()
        model.fit(train[feature_cols], train[self.equipo_col])
        
        self.models[f'lag_{n_lags}'] = {
            'model': model,
            'features': feature_cols,
            'train': train,
            'test': test
        }
        
        return self
    
    def fit_seasonal_model(self):
        """Modelo con componentes estacionales"""
        data = self.prepare_data(n_lags=2)
        train, test, _ = self.train_test_split(data)
        
        feature_cols = (['t', 'lag_1', 'lag_2'] + 
                       [col for col in train.columns if col.startswith('month_')])
        
        model = LinearRegression()
        model.fit(train[feature_cols], train[self.equipo_col])
        
        self.models['seasonal'] = {
            'model': model,
            'features': feature_cols,
            'train': train,
            'test': test
        }
        
        return self
    
    def evaluate_models(self):
        """Evalúa todos los modelos"""
        results = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            features = model_info['features']
            test = model_info['test']
            
            y_pred = model.predict(test[features])
            y_true = test[self.equipo_col]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            results.append({
                'Modelo': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Predicciones': y_pred,
                'Fechas': test['Date'].values
            })
            
            self.results[name] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'dates': test['Date'].values,
                'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            }
        
        return pd.DataFrame(results)
    
    def predict_future(self, model_name, n_periods=36):
        """Predice valores futuros"""
        model_info = self.models[model_name]
        model = model_info['model']
        features = model_info['features']
        
        print(f"Generando predicciones con modelo: {model_name}")
        
        # Obtener datos históricos
        historical = self.df[self.equipo_col].values
        last_values = list(historical[-10:])
        
        # Crear lista para predicciones
        predictions = []
        
        for i in range(n_periods):
            # Construir features
            feat_dict = {}
            
            # Tiempo
            feat_dict['t'] = len(historical) + i
            
            # Lags
            if 'lag_1' in features:
                if i == 0:
                    feat_dict['lag_1'] = last_values[-1]
                else:
                    feat_dict['lag_1'] = predictions[i-1]
            
            if 'lag_2' in features:
                if i == 0:
                    feat_dict['lag_2'] = last_values[-2] if len(last_values) >= 2 else last_values[-1]
                elif i == 1:
                    feat_dict['lag_2'] = predictions[0]
                else:
                    feat_dict['lag_2'] = predictions[i-2]
            
            if 'lag_3' in features:
                if i == 0:
                    feat_dict['lag_3'] = last_values[-3] if len(last_values) >= 3 else last_values[-1]
                elif i == 1:
                    feat_dict['lag_3'] = last_values[-2] if len(last_values) >= 2 else last_values[-1]
                elif i == 2:
                    feat_dict['lag_3'] = predictions[0]
                else:
                    feat_dict['lag_3'] = predictions[i-3]
            
            # Mes (para estacionalidad)
            last_date = self.df['Date'].iloc[-1]
            future_date = last_date + pd.DateOffset(months=i+1)
            current_month = future_date.month
            
            # Dummies de meses
            for feat in features:
                if feat.startswith('month_'):
                    month_num = int(feat.split('_')[1])
                    feat_dict[feat] = 1 if month_num == current_month else 0
            
            # Crear DataFrame
            df_pred = pd.DataFrame([feat_dict])
            
            # Asegurar columnas faltantes
            for feat in features:
                if feat not in df_pred.columns:
                    df_pred[feat] = 0
            
            # Reordenar
            df_pred = df_pred[features]
            
            # Predecir
            pred = model.predict(df_pred)[0]
            predictions.append(pred)
        
        print(f"{len(predictions)} Predicciones generadas")
        return np.array(predictions)

# ======================================================
# PASO 4 — ENTRENAR MODELOS
# ======================================================

print("\n" + "="*50)
print("ENTRENANDO MODELOS")
print("="*50)

# Modelos para Equipo 1
predictor1 = TimeSeriesPredictor(df, 'Equipo1')
predictor1.fit_linear_trend().fit_lag_model(n_lags=3).fit_seasonal_model()
results1 = predictor1.evaluate_models()

print("\nResultados Equipo 1:")
print(results1[['Modelo', 'MAE', 'RMSE', 'R2']])

# Modelos para Equipo 2
predictor2 = TimeSeriesPredictor(df, 'Equipo2')
predictor2.fit_linear_trend().fit_lag_model(n_lags=3).fit_seasonal_model()
results2 = predictor2.evaluate_models()

print("\nResultados Equipo 2:")
print(results2[['Modelo', 'MAE', 'RMSE', 'R2']])

# ======================================================
# PASO 5 — SELECCIONAR MEJORES MODELOS
# ======================================================

best_model1 = results1.loc[results1['RMSE'].idxmin(), 'Modelo']
best_model2 = results2.loc[results2['RMSE'].idxmin(), 'Modelo']

print(f"\nMejor modelo para Equipo 1: {best_model1}")
print(f"Mejor modelo para Equipo 2: {best_model2}")

# ======================================================
# PASO 6 — PREDICCIONES FUTURAS
# ======================================================

print("\n" + "="*50)
print("GENERANDO PREDICCIONES A 36 MESES")
print("="*50)

future_pred1 = predictor1.predict_future(best_model1, n_periods=36)
future_pred2 = predictor2.predict_future(best_model2, n_periods=36)

# ======================================================
# PASO 7 — FECHAS FUTURAS
# ======================================================

ultima_fecha = df['Date'].max()
fechas_futuras = pd.date_range(
    start=ultima_fecha + pd.Timedelta(days=1),
    periods=36,
    freq='ME'
)

print(f"Predicciones desde {fechas_futuras[0].strftime('%Y-%m')} hasta {fechas_futuras[-1].strftime('%Y-%m')}")

# ======================================================
# PASO 8 — CLASE KPI (NUEVO - AÑADIDO AQUÍ)
# ======================================================

print("ANÁLISIS DE KPIS AVANZADOS")

class KPIAnalisis:
    """Clase para análisis de KPIs adicionales"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.resultados = {}
        self._preparar_datos()
        
    def _preparar_datos(self):
        """Prepara datos para análisis de KPIs"""
        self.df['año'] = self.df['Date'].dt.year
        self.df['mes'] = self.df['Date'].dt.month
        self.df['mes_nombre'] = self.df['mes'].apply(lambda x: 
            ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][x-1])
        
    def kpi_riesgo(self):
        """KPI 1: Value at Risk (VAR)"""
        print("\n" + "="*50)
        print("KPI 1: ANÁLISIS DE RIESGO (VAR)")
        print("="*50)
        
        resultados = {}
        for equipo in ['Equipo1', 'Equipo2']:
            var_95 = np.percentile(self.df[equipo].dropna(), 5)
            var_90 = np.percentile(self.df[equipo].dropna(), 10)
            precio_actual = self.df[equipo].iloc[-1]
            colchon = ((precio_actual - var_95) / precio_actual) * 100
            
            resultados[equipo] = {
                'VAR_95': var_95,
                'VAR_90': var_90,
                'colchon_%': colchon
            }
            
            print(f"\n{equipo}:")
            print(f"  • VAR 95%: ${var_95:,.2f} (peor escenario)")
            print(f"  • Colchón recomendado: {colchon:.1f}%")
        
        self.resultados['riesgo'] = resultados
        return resultados
    
    def kpi_estacionalidad(self):
        """KPI 2: Patrones estacionales"""
        print("\n" + "="*50)
        print("KPI 2: ANÁLISIS DE ESTACIONALIDAD")
        print("="*50)
        
        resultados = {}
        for material in ['X', 'Y', 'Z']:
            prom_mensual = self.df.groupby('mes_nombre')[material].mean()
            prom_global = self.df[material].mean()
            
            mejor_mes = prom_mensual.idxmin()
            peor_mes = prom_mensual.idxmax()
            ahorro = (1 - prom_mensual.min()/prom_global) * 100
            
            resultados[material] = {
                'mejor_mes': mejor_mes,
                'peor_mes': peor_mes,
                'ahorro_potencial': ahorro
            }
            
            print(f"\n{material}:")
            print(f"  • Mejor momento: {mejor_mes} (ahorro {ahorro:.1f}%)")
            print(f"  • Evitar: {peor_mes}")
        
        self.resultados['estacionalidad'] = resultados
        return resultados
    
    def kpi_correlacion(self):
        """KPI 3: Matriz de correlación"""
        print("\n" + "="*50)
        print("KPI 3: ANÁLISIS DE CORRELACIÓN")
        print("="*50)
        
        corr_matrix = self.df[['X', 'Y', 'Z']].corr()
        
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, fmt='.3f')
        plt.title('Matriz de Correlación - Materias Primas')
        plt.tight_layout()
        plt.savefig('Resultados/correlacion_kpi.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("\nInterpretación:")
        for i in ['X', 'Y', 'Z']:
            for j in ['X', 'Y', 'Z']:
                if i < j:
                    corr = corr_matrix.loc[i, j]
                    nivel = "fuerte" if abs(corr) > 0.7 else "moderada" if abs(corr) > 0.4 else "débil"
                    print(f"  • Correlación {nivel} entre {i} y {j}: {corr:.3f}")
        
        self.resultados['correlacion'] = corr_matrix
        return corr_matrix
    
    def kpi_eficiencia(self):
        """KPI 4: Índice de eficiencia de compra"""
        print("\n" + "="*50)
        print("KPI 4: EFICIENCIA DE COMPRA")
        print("="*50)
        
        ventana = 30
        eficiencia_actual = {}
        
        for material in ['X', 'Y', 'Z']:
            precio_actual = self.df[material].iloc[-1]
            promedio_historico = self.df[material].mean()
            eficiencia = precio_actual / promedio_historico
            
            if eficiencia < 1:
                estado = "Excelente momento"
            elif eficiencia < 1.05:
                estado = "Precio justo"
            else:
                estado = "Esperar mejora"
            
            eficiencia_actual[material] = {
                'precio_actual': precio_actual,
                'eficiencia': eficiencia,
                'estado': estado
            }
            
            print(f"\n{material}:")
            print(f"  • Precio actual: ${precio_actual:.2f}")
            print(f"  • {estado}")
        
        self.resultados['eficiencia'] = eficiencia_actual
        return eficiencia_actual
    
    def kpi_tendencia_anual(self):
        """KPI 5: Tendencia anual de costos"""
        print("\n" + "="*50)
        print("KPI 5: TENDENCIA ANUAL")
        print("="*50)
        
        tendencias = self.df.groupby('año')[['Equipo1', 'Equipo2']].mean()
        
        print("\nEvolución anual de costos:")
        print(tendencias.round(2))
        
        # Calcular crecimiento
        crecimiento_eq1 = ((tendencias['Equipo1'].iloc[-1] / tendencias['Equipo1'].iloc[0]) - 1) * 100
        crecimiento_eq2 = ((tendencias['Equipo2'].iloc[-1] / tendencias['Equipo2'].iloc[0]) - 1) * 100
        
        print(f"\nCrecimiento en el período:")
        print(f"  • Equipo 1: {crecimiento_eq1:.1f}%")
        print(f"  • Equipo 2: {crecimiento_eq2:.1f}%")
        
        return tendencias
    
    def kpi_resumen_ejecutivo(self):
        """KPI 6: Dashboard ejecutivo"""
        print("\n" + "="*50)
        print("KPI 6: RESUMEN EJECUTIVO")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Evolución de equipos
        axes[0,0].plot(self.df['Date'], self.df['Equipo1'], label='Equipo 1', color='red', alpha=0.7)
        axes[0,0].plot(self.df['Date'], self.df['Equipo2'], label='Equipo 2', color='blue', alpha=0.7)
        axes[0,0].set_title('Evolución Histórica')
        axes[0,0].set_ylabel('Costo ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Boxplot comparativo
        data_box = [self.df['Equipo1'].dropna(), self.df['Equipo2'].dropna()]
        bp = axes[0,1].boxplot(data_box, labels=['Equipo 1', 'Equipo 2'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('blue')
        axes[0,1].set_title('Variabilidad')
        axes[0,1].set_ylabel('Costo ($)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Estacionalidad
        meses = ['E', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        for material in ['X', 'Y', 'Z']:
            prom_mensual = self.df.groupby('mes')[material].mean()
            axes[1,0].plot(meses, prom_mensual.values, marker='o', label=material)
        axes[1,0].set_title('Patrón Estacional')
        axes[1,0].set_xlabel('Mes')
        axes[1,0].set_ylabel('Precio ($)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Métricas clave
        axes[1,1].axis('off')
        
        # Obtener métricas
        riesgo_eq2 = self.resultados.get('riesgo', {}).get('Equipo2', {}).get('colchon_%', 0)
        mejor_mes_x = self.resultados.get('estacionalidad', {}).get('X', {}).get('mejor_mes', 'N/A')
        ahorro_x = self.resultados.get('estacionalidad', {}).get('X', {}).get('ahorro_potencial', 0)
        
        texto = f"""
        HALLAZGOS PRINCIPALES:
        
        1. RIESGO:
           • Equipo 1: más estable
           • Equipo 2: requiere {riesgo_eq2:.1f}% colchón
           
        2. AHORRO POTENCIAL:
           • Mejor mes para X: {mejor_mes_x}
           • Ahorro: {ahorro_x:.1f}%
           
        3. MOMENTO ACTUAL:
           • X: {self.resultados.get('eficiencia', {}).get('X', {}).get('estado', 'N/A')}
           • Y: {self.resultados.get('eficiencia', {}).get('Y', {}).get('estado', 'N/A')}
           • Z: {self.resultados.get('eficiencia', {}).get('Z', {}).get('estado', 'N/A')}
        """
        
        axes[1,1].text(0.1, 0.9, texto, fontsize=10, verticalalignment='top',
                      fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('Resultados/dashboard_kpis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Dashboard guardado en 'Resultados/dashboard_kpis.png'")
    
    def ejecutar_todos_kpis(self):
        """Ejecuta todos los KPIs en secuencia"""
        self.kpi_riesgo()
        self.kpi_estacionalidad()
        self.kpi_correlacion()
        self.kpi_eficiencia()
        self.kpi_tendencia_anual()
        self.kpi_resumen_ejecutivo()
        
        # Guardar resultados
        with open('Resultados/kpis_resultados.txt', 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DE KPIS\n")
            f.write("="*50 + "\n")
            f.write(str(self.resultados))
        
        return self.resultados

# ======================================================
# PASO 9 — EJECUTAR KPIS
# ======================================================

print("EJECUTANDO ANÁLISIS DE KPIS")

analisis_kpis = KPIAnalisis(df)
resultados_kpis = analisis_kpis.ejecutar_todos_kpis()

# ======================================================
# PASO 10 — VISUALIZACIÓN PREDICCIONES
# ======================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Equipo 1
axes[0].plot(df['Date'], df['Equipo1'], label='Histórico', alpha=0.7)
axes[0].plot(fechas_futuras, future_pred1, label=f'Predicción {best_model1}', linewidth=2, color='red')
axes[0].axvline(x=ultima_fecha, color='black', linestyle='--', alpha=0.5)
axes[0].set_title(f'Equipo 1 - Predicción 36 meses')
axes[0].set_ylabel('Costo')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Equipo 2
axes[1].plot(df['Date'], df['Equipo2'], label='Histórico', alpha=0.7)
axes[1].plot(fechas_futuras, future_pred2, label=f'Predicción {best_model2}', linewidth=2, color='red')
axes[1].axvline(x=ultima_fecha, color='black', linestyle='--', alpha=0.5)
axes[1].set_title(f'Equipo 2 - Predicción 36 meses')
axes[1].set_ylabel('Costo')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Resultados/predicciones_finales.png', dpi=150)
plt.show()

# ======================================================
# PASO 11 — CÁLCULO DE COSTOS TOTALES
# ======================================================

costo_total_eq1 = future_pred1.sum()
costo_total_eq2 = future_pred2.sum()

print("\n" + "="*50)
print("COSTOS ESTIMADOS DEL PROYECTO (36 MESES)")
print("="*50)
print(f"Equipo 1: ${costo_total_eq1:,.2f}")
print(f"Equipo 2: ${costo_total_eq2:,.2f}")
print(f"Total Proyecto: ${costo_total_eq1 + costo_total_eq2:,.2f}")

# ======================================================
# PASO 12 — GUARDAR RESULTADOS FINALES
# ======================================================

resultados_finales = pd.DataFrame({
    'Métrica': ['Mejor Modelo Equipo 1', 'Mejor Modelo Equipo 2', 
                'Costo Total Equipo 1', 'Costo Total Equipo 2', 'Costo Total Proyecto'],
    'Valor': [best_model1, best_model2, 
              f"${costo_total_eq1:,.2f}", f"${costo_total_eq2:,.2f}", 
              f"${costo_total_eq1 + costo_total_eq2:,.2f}"]
})

resultados_finales.to_csv('Resultados/resumen_final.csv', index=False)

print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("Revisa la carpeta 'Resultados'")

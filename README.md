# MVP-Machine-Learning

Descripción
-----------
Este repositorio contiene un MVP de Machine Learning con dos archivos en la raíz:
- `MVP.py` — posible aplicación o script principal.
- `main.ipynb` — notebook con exploración / entrenamiento / demostraciones.

Si tu intención es servir una interfaz con Streamlit, es habitual que `MVP.py` sea la app. Si no, este README explica cómo comprobarlo y cómo ejecutar ambos archivos.

Requisitos
----------
- Python 3.8+
- pip
- (opcional) virtualenv o conda

Instalación rápida
------------------
1. Clona el repositorio:
```bash
git clone https://github.com/Gelviz274/MVP-Machine-Learning.git
cd MVP-Machine-Learning
```

2. (Recomendado) Crea y activa un entorno virtual:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Instala dependencias mínimas (si no hay `requirements.txt`):
```bash
pip install streamlit pandas numpy scikit-learn jupyter
```
Si luego creas `requirements.txt`, usa:
```bash
pip install -r requirements.txt
```

Comprobar si `MVP.py` es una app de Streamlit
---------------------------------------------
Abre `MVP.py` o busca la importación de Streamlit. Desde la terminal:
```bash
grep -n "import streamlit" MVP.py || grep -n "streamlit" MVP.py
```
- Si aparece `import streamlit as st` (u otra referencia a `streamlit`), `MVP.py` muy probablemente es una app de Streamlit.
- Si no, podría ser un script de entrenamiento u otro tipo de utilidad.

Cómo ejecutar (opciones)
------------------------
1. Si `MVP.py` es una app de Streamlit:
```bash
streamlit run MVP.py
```
Por defecto se abre en: http://localhost:8501

Si necesitas cambiar puerto o dirección:
```bash
streamlit run MVP.py --server.port 8501 --server.address 0.0.0.0
```

2. Si `MVP.py` es un script Python normal (no Streamlit):
```bash
python MVP.py
```
Lee la salida o el código para ver parámetros requeridos.

3. Para abrir el notebook:
```bash
jupyter notebook main.ipynb
# o si usas JupyterLab
jupyter lab main.ipynb
```

Sugerencias y buenas prácticas
------------------------------
- Añade un `requirements.txt` con versiones concretas.
- Añade un `.gitignore` que excluya entornos virtuales, data grandes y modelos.
- Si `MVP.py` es la app Streamlit, documenta en README la URL y el comando exacto.
- Si hay datos grandes, añade instrucciones para descargarlos o un script de setup.
- Considera dividir código en `src/` y mantener `MVP.py` como launcher/entrypoint.

¿Te ayudo a confirmar si `MVP.py` usa Streamlit y a actualizar este README con el comando exacto? Puedo abrir `MVP.py` ahora y devolver el resultado. 

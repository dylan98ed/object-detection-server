# 📌 Intel RealSense Streaming Server + YOLO en Jetson Orin Nano

Este repositorio contiene un servidor basado en Flask que transmite en vivo el video de una cámara **Intel RealSense D435**, mostrando:
- **Stream RGB** (imagen a color)
- **Stream de Profundidad** (mapa de profundidad con escala de colores)
- **Stream procesado con YOLO** (detección de objetos en tiempo real con un modelo entrenado)

## 🚀 Características
- **Transmisión en vivo** del video de la cámara RealSense.
- **Tres streams en paralelo**: RGB, Profundidad y Procesamiento con YOLO.
- **Compatibilidad con la Jetson Orin Nano**, aprovechando aceleración por hardware.
- **Interfaz web accesible** desde cualquier dispositivo en la misma red.
- **Monitoreo del rendimiento de la Jetson** con `jtop`.

## 🛠 Tecnologías utilizadas
- **Python 3**
- **Flask** (para el servidor web)
- **OpenCV** (para procesamiento de imágenes)
- **Intel RealSense SDK (pyrealsense2)** (para obtener frames de la cámara)
- **Ultralytics YOLO** (para detección de objetos en tiempo real)
- **NVIDIA Jetson SDK** (para optimización en la Jetson Orin Nano)

## 📦 Instalación de dependencias
Antes de ejecutar el script, asegúrate de instalar todas las dependencias necesarias.

### 1️⃣ Instalar los paquetes necesarios en Jetson Orin Nano
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip libgl1 -y
```

### 2️⃣ Instalar las bibliotecas de Python
```bash
pip3 install flask opencv-python numpy pyrealsense2 ultralytics
```

### 3️⃣ Instalar el SDK de Intel RealSense
```bash
sudo apt install librealsense2-utils librealsense2-dev
```
Verifica la conexión de la cámara con:
```bash
rs-enumerate-devices
```
Si la cámara está bien conectada, se mostrará la información del dispositivo.

### 4️⃣ Instalar `jtop` para monitorear el rendimiento de la Jetson
```bash
sudo apt install python3-pip
pip3 install jetson-stats
```
Para visualizar el consumo de recursos de la Jetson, ejecuta:
```bash
jtop
```

## 🚀 Ejecución del servidor
Clona este repositorio y navega al directorio:
```bash
git clone https://github.com/dylan98ed/jetson-realsense-streaming.git
cd jetson-realsense-streaming
```
Luego, ejecuta el servidor:
```bash
python3 realsense_streaming.py
```

## 🌐 Acceder al streaming
Desde cualquier dispositivo en la misma red, abre un navegador y accede a:
```
http://<IP_DE_LA_JETSON>:5000
```
Ejemplo:
```
http://192.168.1.100:5000
```
Aquí verás tres streams en paralelo:
- **RGB Stream**
- **Depth Stream**
- **Processed Stream (YOLO)**

## 🖥 Interfaz Web
La interfaz muestra los tres streams en paralelo:
```
+----------------+  +----------------+  +----------------+
|  RGB Stream   |  | Depth Stream   |  | YOLO Stream   |
|  (Video)      |  | (Colormap)     |  | (Detección)   |
+----------------+  +----------------+  +----------------+
```

## 📜 Licencia
Este proyecto es de código abierto y puedes usarlo, modificarlo y compartirlo libremente.

Si encuentras útil este código, considera darle una estrella ⭐ en GitHub. ¡Gracias! 🚀


# Object Detection Aplication

Aplicación de reconocimiento de objetos en tiempo real que utiliza la API de detección de objetos TensorFlow de Google y OpenCV .


## Requisitos

Algunos de los requisitos fundamentales para la ejecución de este script son los siguientes:

* Python 3.6.5
* TensorFlow with GPU support V. 1.9.0 -> https://www.tensorflow.org/install/install_windows
* CUDA Toolkit 9.0.
* The NVIDIA drivers associated with CUDA Toolkit 9.0.
* cuDNN v7.0.
* GPU card

## Composición del proyecto 

Este proyecto esta formado por los siguientes archivos:

* Data ->Carpeta donde se incluyen los puntos pbtxt donde se incluyen las distintas etiquetas de los modelos que estemos usando
* utils ->Carpeta donde se incluyen dos archivos python para pintar en timepo real las deteciones en el video capturado por la cámara
* real_time_streaming_filter ->Archivo donde se encuentra toda la lógica ya comentada en el propio script
* Modelos prueba

## Modelos ejemplo adjuntados

En este proyecto se adjunta dos modelos de prueba previamente entrenados:

* frozen_inference_graph.pb -> Detector de bastones
* frozen_inference_graph_normal.pb ->Detector de personas entre otros objetos

Ambos se puede sustituir por cualquier otro con otras funcionalidades

## Ejecución del proyecto

Para la ejecución del proyecto se lanzará el siguiente comando:
```
python real_time_streaming_filter.py

```
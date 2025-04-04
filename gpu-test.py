import torch
print(torch.cuda.is_available())  # Debe devolver True
print(torch.cuda.device_count())  # Debe devolver un número mayor a 0
print(torch.cuda.get_device_name(0))  # Verifica qué GPU está detectando
print(torch.version.cuda)  # Debe mostrar 12.6 si instalaste la versión correcta
print(torch.backends.cudnn.version())  # Verifica si cuDNN está instalado correctamente

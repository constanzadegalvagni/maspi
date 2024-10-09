import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

# Cargar y transformar la imagen
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises
        transforms.ToTensor()    # Convertir a tensor
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir un batch dimension
    return image

# Definir filtros de Prewitt
prewitt_x = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
prewitt_y = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Aplicar el filtro de convolución
def apply_prewitt_filter(image, kernel):
    return F.conv2d(image, kernel, padding=1)

# Cargar imagen
image_path = '/home/constanza/facu/metodos avanzados de sintesis y procesamiento de imagenes/taller1/fig3.jpg'
image = load_image(image_path)

# Aplicar filtros de Prewitt
edges_x = apply_prewitt_filter(image, prewitt_x)
edges_y = apply_prewitt_filter(image, prewitt_y)

# Calcular la magnitud de los bordes
edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

# Implementamos filtro diagonal
diagonal_x_y = torch.tensor([[1, 2, 0], [2, 0, -2], [0, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
diagonal_x_menosy = torch.tensor([[0, -2, -1], [2, 0, -2], [1, 2, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

diagonal_x_y_viejo = torch.tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
diagonal_x_menosy_viejo = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#diagonal_x_y_5 = torch.tensor([[-4,-3,-2,-1,0], [-3,-2,-1,0,1], [-2,-1,0,1,2], [-1,0,1,2,3],[0,1,2,3,4]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#diagonal_x_menosy_5 = torch.tensor([[0,-1,-2,-3,-4], [-3,-2,-1,0,1], [-2,-1,0,1,2],[-1,0,1,2,3],[0,1,2,3,4]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

diagonal_x_y_5 = torch.tensor([[-1,-3/4,-1/2,-1/4,0], [-3/4,-1/2,-1/4,0,1/4], [-1/2,-1/4,0,1/4,1/2], [-1/4,0,1/4,1/2,3/4],[0,1/4,1/2,3/4,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
diagonal_x_menosy_5 = torch.tensor(np.array([[-1,-3/4,-1/2,-1/4,0], [-3/4,-1/2,-1/4,0,1/4], [-1/2,-1/4,0,1/4,1/2], [-1/4,0,1/4,1/2,3/4],[0,1/4,1/2,3/4,1]]).T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

diag_y_viejo = apply_prewitt_filter(image,diagonal_x_y_viejo)
diag_menosy_viejo = apply_prewitt_filter(image,diagonal_x_menosy_viejo)

diag_y = apply_prewitt_filter(image,diagonal_x_y)
diag_menosy = apply_prewitt_filter(image,diagonal_x_menosy)

diag_y_5 = apply_prewitt_filter(image,diagonal_x_y_5)
diag_menosy_5 = apply_prewitt_filter(image,diagonal_x_menosy_5)

diag = torch.sqrt(diag_y**2 + diag_menosy ** 2)
diag_5 = torch.sqrt(diag_y_5**2 + diag_menosy_5 **2)
diag_viejo = torch.sqrt(diag_y_viejo**2 + diag_menosy_viejo ** 2)

""" # Visualizar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')

plt.subplot(1, 4, 3)
plt.title('Filtro Prewitt Horizontal')
plt.imshow(edges_x.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 4, 4)
plt.title('Filtro Prewitt Vertical')
plt.imshow(edges_y.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 4, 2)
plt.title('Modulo de los gradientes')
plt.imshow(edges.squeeze().detach().numpy(), cmap='gray')

plt.show() """

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Filtro Diagonal x=y')
plt.imshow(diag_y.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Filtro Diagonal $x = -y$')
plt.imshow(diag_menosy.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Modulo de los gradientes')
plt.imshow(diag.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.title("3x3")
plt.show(block=False)

""" plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Filtro Diagonal x=y')
plt.imshow(diag_y_5.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Filtro Diagonal $x = -y$')
plt.imshow(diag_menosy_5.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Modulo de los gradientes')
plt.imshow(diag_5.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.title("5x5")
plt.show() """

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Filtro Diagonal x=y')
plt.imshow(diag_y_viejo.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Filtro Diagonal $x = -y$')
plt.imshow(diag_menosy_viejo.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Modulo de los gradientes')
plt.imshow(diag_viejo.squeeze().detach().numpy(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.title("3x3 viejo")
plt.show()
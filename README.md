# 🎨 Image Generation using Variational Autoencoders (VAE)

A Deep Learning project that generates new images using Variational Autoencoders (VAE) trained on the CIFAR-10 dataset.

This project demonstrates:

- Variational Autoencoders (VAE)
- Conditional Variational Autoencoders (CVAE)
- Latent Space Learning
- Image Generation
- Latent Space Interpolation
- t-SNE Latent Visualization
- Comparison between VAE and GANs

---

# 🚀 Project Features

✅ Image Generation using VAE  
✅ CIFAR-10 Dataset Training  
✅ CNN-based Encoder & Decoder  
✅ Reparameterization Trick  
✅ Latent Space Visualization  
✅ Latent Vector Interpolation  
✅ Conditional VAE (Class Controlled Generation)  
✅ Deep Learning Visualization  
✅ Generative AI Concepts  

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning Framework |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |
| scikit-learn | t-SNE Visualization |

---

# 📂 Project Structure

```bash
vae_project/
│
├── train_vae.py
├── latent_interpolation.py
├── visualize_latent.py
├── conditional_vae.py
│
├── models/
│   ├── encoder.keras
│   └── decoder.keras
│
├── outputs/
│   ├── generated_images.png
│   ├── interpolation.png
│   └── latent_space.png
│
└── README.md
```

---

# 📊 Dataset

This project uses the CIFAR-10 dataset.

Dataset Details:

- 60,000 RGB images
- 10 classes
- Image size: 32×32
- Classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

---

# 🏗️ Variational Autoencoder Architecture

```text
Input Image
     ↓
Encoder (CNN)
     ↓
Latent Distribution
(μ and σ)
     ↓
Reparameterization
     ↓
Latent Vector z
     ↓
Decoder (CNN)
     ↓
Generated Image
```

---
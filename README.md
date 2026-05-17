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

# 🔥 Key Deep Learning Concepts

## 1. Encoder

The encoder compresses input images into a latent representation.

---

## 2. Decoder

The decoder reconstructs images from latent vectors.

---

## 3. Latent Space

Latent space is a compressed mathematical representation of images learned by the model.

---

## 4. Reparameterization Trick

VAE uses:

\[
z = \mu + \sigma \epsilon
\]

where:

\[
\epsilon \sim N(0,1)
\]

This allows backpropagation through stochastic sampling.

---

## 5. KL Divergence

KL Divergence regularizes latent space to follow Gaussian distribution.

---

# 📉 Loss Function

Total VAE Loss:

\[
L = L_{reconstruction} + D_{KL}
\]

Where:

- Reconstruction Loss → Measures image reconstruction quality
- KL Divergence → Regularizes latent distribution

---

# 🎯 Conditional Variational Autoencoder (CVAE)

CVAE generates images conditioned on class labels.

Example:

- Generate only dogs
- Generate only ships
- Generate only trucks

Flow:

```text
Image + Label
      ↓
Encoder
      ↓
Latent Space
      ↓
Decoder + Label
      ↓
Controlled Image Generation
```

---

# 🌌 Latent Space Interpolation

Interpolation smoothly transitions between two latent vectors.

Formula:

\[
z = (1-\alpha)z_1 + \alpha z_2
\]

This demonstrates continuity of latent space.

---

# 📈 Latent Space Visualization

t-SNE is used to visualize high-dimensional latent vectors in 2D space.

This helps observe clustering of similar image classes.

---

# ⚔️ VAE vs GAN

| VAE | GAN |
|---|---|
| Stable Training | Difficult Training |
| Blurry Images | Sharper Images |
| Probabilistic | Adversarial |
| Better Latent Space | Better Image Quality |

---

# ▶️ How to Run the Project

## 1️⃣ Install Dependencies

```bash
pip install tensorflow matplotlib numpy scikit-learn
```

---

## 2️⃣ Train VAE

```bash
python train_vae.py
```

---

## 3️⃣ Run Latent Interpolation

```bash
python latent_interpolation.py
```

---

## 4️⃣ Visualize Latent Space

```bash
python visualize_latent.py
```

---

## 5️⃣ Run Conditional VAE

```bash
python conditional_vae.py
```

---

# 📷 Expected Outputs

The project generates:

- CIFAR-like synthetic images
- Latent interpolation transitions
- Latent cluster visualization
- Class-specific generated images


---

# 💡 Applications

- AI Image Generation
- Deepfake Systems
- Data Augmentation
- Generative AI
- Medical Image Synthesis
- Representation Learning

---

# 👨‍💻 Author

Developed as a Deep Learning practical project using Variational Autoencoders on CIFAR-10 dataset.

---


# ❤️ Thank You

If you found this project useful, consider giving it a ⭐ on GitHub.
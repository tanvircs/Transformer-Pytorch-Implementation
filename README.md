# 🔥 Transformer Self-Attention & Masked Self-Attention (PyTorch)

This repository implements **Self-Attention** and **Masked Self-Attention** in PyTorch, essential components of the **Transformer architecture** (Vaswani et al., 2017). 🚀

- ✅ Implemented **Self-Attention Mechanism**
- ✅ Implemented **Masked Self-Attention (for causal attention)**
- ✅ Included **matrix operations for Q, K, V computations**
- ✅ Supports **custom input encodings and masking**
- ✅ Written in **PyTorch** for easy extensibility

---

## 🔍 **Understanding Self-Attention**

Self-attention allows a model to weigh different parts of the input sequence dynamically.

### **👉 Self-Attention Formula:**

\[
\text{Attention}(Q, K, V) = \text{softmax} \left(\frac{QK^T}{\sqrt{d\_{\text{model}}}} \right) V
\]

Each input token generates:

- **Query (Q)** → Represents what it is looking for.
- **Key (K)** → Represents what it contains.
- **Value (V)** → Represents actual token embeddings.

---

## 🛠 **Implementation Details**

This repository contains two key implementations:

### **1️⃣ Self-Attention (`Transformer_SelfAttention`)**

📌 Computes **attention scores** over the entire sequence.

#### **Code Example**

```python
tsa = Transformer_SelfAttention(dimentiona_model=2, row_dimension=0, column_dimension=1)
tsa(encoding_matrix)
```

### **2️⃣ Masked Self-Attention (`Transformer_MaskSelfAttention`)**
📌 Used in decoder models (GPT, Transformer Decoder) to prevent attention to future tokens.

---

## 🔒 **Causal Masking**

A lower-triangular mask is used:

```python
[[False,  True,  True],
 [False, False,  True],
 [False, False, False]]
```

### **Meaning:**
- **Token 1** attends only to itself.
- **Token 2** attends to itself + previous tokens.
- **Token 3** attends to itself + all previous tokens.

#### **Code Example**

```python
mask = torch.tril(torch.ones(3,3)) == 0  # Lower triangular mask
tmsat(encoding_matrix, mask)  # Apply masked self-attention
```

---

## 📂 **Project Structure**

```bash
📦 Transformer-SelfAttention
├── 📜 self_attention.py          # Self-Attention Implementation
├── 📜 masked_self_attention.py   # Masked Self-Attention Implementation
├── 📜 demo.ipynb                 # Jupyter Notebook with Examples
├── 📜 README.md                  # Documentation
```

---

## 🚀 **Run the Code**

### 📌 Install Dependencies

```bash
pip install torch numpy
```

### 📌 Run Self-Attention

```bash
python self_attention.py
```

### 📌 Run Masked Self-Attention

```bash
python masked_self_attention.py
```

---

## 📊 **Results & Visualizations**

Self-Attention and Masked Self-Attention results include:

- Query, Key, Value transformations
- Similarity Scores
- Attention Scores
- Masking Effect (Masked Self-Attention)

### **Example output:**

```python
Similarity Scores:
tensor([[ 2.35, -0.56,  4.12],
        [ 1.89,  3.44, -1.23],
        [-0.23,  1.78,  5.89]])

Scaled Similarity Scores (after sqrt(d_model)):
tensor([[ 1.67, -0.40,  2.91],
        [ 1.34,  2.43, -0.87],
        [-0.16,  1.26,  4.18]])

Attention Scores:
tensor([[ 1.92,  0.47],
        [ 2.15, -0.65],
        [-1.78,  3.05]])
```


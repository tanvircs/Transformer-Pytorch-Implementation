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

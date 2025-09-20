# CNN Concepts & Q&A Reference

This README summarizes key Convolutional Neural Network (CNN) concepts, layer calculations, and reasoning from the Q&A session. It is intended as a **quick reference for CNN architecture, receptive field, pooling, kernel parameters, and GPU memory considerations**.

---

## 1. Convolution Output Size

**Q:** If we perform a convolution with a kernel of size 3x3 on a 47x49 image, what is the output size?

**A:** 45x47

**Reasoning:**

$$
\text{Output size} = \text{Input size} - \text{Kernel size} + 1
$$

- Height: 47 − 3 + 1 = 45
- Width: 49 − 3 + 1 = 47

---

## 2. CNN Layer Properties

**True Statements:**

- Normally add **padding** to keep output size same as input (same padding)

- Nearly always use kernels with **stride 1** unless pooling is applied

- Add layers to reach **full receptive field**

- We always use 3x3 kernels — 3x3 is common but not mandatory

---

## 3. Receptive Field Calculation

**Q:** How many 3x3 convolution layers are needed to reach a receptive field of 21x21?

**A:** 10

**Reasoning:**

$$
R = 1 + n \cdot (k-1), \quad k=3
$$

$$
21 = 1 + n \cdot 2 \implies n=10
$$

---

## 4. Let us assume we have an image of size 100x100. What is the minimum number of **convolution** **layers** do we need to add such that 

**Scenario:**&#x20;

1. you cannot use max-pooling without convolving twice or more
2. the output is at least 2-3 convolution layers away from max-pooling
3. You can stop either at 2x2 or 1x1 based on how you have used your layers
4. we will always "not consider" the last rows and columns in an odd-resolution channel while performing max-pooling)
5. "do not" count the max-pooling layer
6. do not add padding
7. do not use convolutions with strides of more than 1



**A:** 10 convolution layers

**Reasoning:**

- Must have ≥2 convs before each max-pooling
- Must be ≥2 convs away from last max-pool
- Example schedule: 2+2+2+2 = 10 convs

---

## 5. Kernel Count for a Layer

**Scenario:** If the input layer has 128 channels, how many kernels do we need to add?

**A:** Number of kernels **does not depend on input channels**

**Reasoning:**

- Each kernel spans all input channels
- Number of kernels determines **output channels**, which is a design choice

---

## 6. Kernel Parameters Calculation

**Scenario:** 49x49x256 convolved with 512 kernels of size 3x3

**A:** 1,179,648 parameters

**Reasoning:**

- Parameters per kernel = 256 × 3 × 3 = 2,304
- Total parameters = 2,304 × 512 = 1,179,648

---

## 7. Channels in GPU RAM Before Max-Pooling

**Scenario:** Network with multiple conv layers, before a max-pooling layer.

**A:** 2016 channels

**Reasoning:**

- Layers with output >350x350: 3+32+64+128+256+512+1024 = 2019

### Table: Feature Map Channels

| Layer                 | Output Size (HxW) | # Channels | Channels >350x350? | Cumulative Channels >350x350 |
| --------------------- | ----------------- | ---------- | ------------------ | ---------------------------- |
| Input Image           | 400x400           | 3          | ✅                  | 3                            |
| Conv1: 32x(3x3x3)     | 398x398           | 32         | ✅                  | 35                           |
| Conv2: 64x(3x3x32)    | 396x396           | 64         | ✅                  | 99                           |
| Conv3: 128x(3x3x64)   | 394x394           | 128        | ✅                  | 227                          |
| Conv4: 256x(3x3x128)  | 392x392           | 256        | ✅                  | 483                          |
| Conv5: 512x(3x3x256)  | 390x390           | 512        | ✅                  | 995                          |
| Conv6: 1024x(3x3x512) | 388x388           | 1024       | ✅                  | 2019                         |
| MaxPooling 2x2        | 194x194           | 1024       | ✅                  | 2019                         |



---

## 8. Advantages of Max-Pooling

- ✅ Reduces spatial dimensions (HxW)
- ✅ Provides slight **translational invariance**
- ❌ Does **not** reduce number of channels
- ❌ Does **not** provide rotational invariance

---

## 9. Information Loss Through Pooling

**Scenario:** 400x400 input → 50x50 after 4x max-pooling, 1000 channels

**A:** No, convolution + pooling **filter information**, they do not simply discard it

**Reasoning:**

- Depth expansion (3 → 1000) preserves salient features
- Raw pixel counting (e.g., “64x loss”) is misleading
- Network retains task-relevant information while discarding redundancy

---

## 10. Key Takeaways

- **Receptive field:** grows by (k-1) per conv layer
- **Kernel parameters:** (# input channels × kernel size × kernel size) × # kernels
- **Max-pooling:** reduces spatial size, not channel depth; provides translational invariance
- **Information retention:** Conv + pooling layers **re-encode** rather than strictly lose information
- **GPU memory:** intermediate feature maps occupy channels × spatial dimensions

---

This README is a **comprehensive reference** for CNN design, calculations, and memory considerations.


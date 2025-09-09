# Myntra Product Classification with FastText

This project demonstrates how to **train a supervised FastText model** to classify products from Myntra into categories based on their names, tags, and brands.  

We use the Myntra dataset from [datasets repo](https://github.com/AshishJangra27/datasets) and train a FastText model for **multi-class classification**.

---

## 📌 Requirements
Install the dependencies:
```python
pip install fasttext pandas numpy
```
---

## 📂 Dataset
We use:
```bash
datasets/Myntra Products/products.csv
```
Columns used:

- product_name → Name of the product
- product_tag → Category tag
- brand_tag → Brand
- tags → Labels for classification

---

## 🚀 Steps
1. Clone Dataset
    ```python
    !git clone https://github.com/AshishJangra27/datasets
    import pandas as pd
    
    df = pd.read_csv('/content/datasets/Myntra Products/products.csv')
    df.head(2)
    ```
2. Data Preprocessing
   - Create features = product_name + product_tag + brand_tag
   - Convert tags into FastText label format
    ```python
    df['features'] = df['product_name'] + ' ' + df['product_tag'] + ' ' + df['brand_tag']
    df['tags'] = ',' + df['tags']
    df['label'] = [' __label__'.join(tag.split(',')).strip().lower() for tag in df['tags']]
    
    # Prepare final training data
    data = [i[-1] + ' ' + i[-2] for i in df.values]
    
    with open('data.txt', 'w') as f:
        for i in data:
            f.write(i + '\n')
    ```
3. Train FastText Model
    ```python
    import fasttext
    model = fasttext.train_supervised(input='data.txt')
    ```
4. NumPy Fix for Predict (Optional for NumPy 2.0)

FastText's default predict has issues with NumPy 2.0.
We patch it:
    
    import numpy as np
    
    def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        if isinstance(text, (list, tuple)):
            all_labels, all_probs = [], []
            for t in text:
                labels, probs = self.predict(t, k=k, threshold=threshold, on_unicode_error=on_unicode_error)
                all_labels.append(labels)
                all_probs.append(probs)
            return all_labels, np.asarray(all_probs)
    
        if not isinstance(text, str):
            text = str(text)
    
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
    
        if predictions is None:
            probs, labels = ([], ())
        else:
            labels, probs = zip(*predictions)
            labels = tuple(labels)
            probs = list(probs)
    
        return labels, np.asarray(probs)
    
    fasttext.FastText._FastText.predict = _patched_predict

5. Test the Model
   ```python
   import numpy as np

    for i in range(5):
        txt = df['features'][np.random.randint(len(df['features']))]
        print("Product:", txt)
        print("Predictions:", model.predict(txt.lower(), k=5)[0])
        print('-'*50)

   ```

✅ Sample Output
  ```bash
  Product: Women Ethnic Motifs Printed Kurta kurtas nayo
  Predictions: ('__label__kurtas', '__label__ethnic-wear', '__label__clothing', '__label__indianwear', '__label__fashion')
  
  Product: Leather White Green Sling Bag handbags da-milano
  Predictions: ('__label__handbags', '__label__accessories', '__label__bags', '__label__leather', '__label__fashion')
  ```
---

## 📂 Repository Structure
```bash
├── datasets/                                      # Contains Myntra dataset
│   └── Myntra Products/
│       └── products.csv
├── Tag_Recommendation_with_FastText.ipynb         # Training & prediction script
├── README.md                                      # Documentation
```
---

## ✨ Author

Rishita Priyadarshini Saraf

mail: rishitasarafp@gmail.com

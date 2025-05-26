# 🔍 DimenSight - High-Dimensional Data Visualizer

![UMAP 3D Projection](assets/umap_plot.png)

**DimenSight** is an advanced Python toolkit that reveals hidden patterns in complex datasets through interactive dimensionality reduction visualizations.

## 🌟 Key Features

- **Multi-Algorithm Support**:
  - PCA (linear dimensionality reduction)
  - t-SNE (non-linear local relationships)
  - UMAP (global/local structure preservation)
- **Flexible Visualization**:
  - 2D & 3D projections
  - Cluster coloring by labels
  - Adjustable point size/opacity
- **Professional Outputs**:
  - Publication-ready figures
  - High-resolution PNG exports
  - Customizable styling
- **Seamless Integration**:
  - Works with NumPy/Pandas data
  - Accepts CSV/Excel inputs
  - Compatible with scikit-learn pipelines

## 🚀 Quick Start

### Installation
```bash
pip install dimensight  # Coming soon!
# Or clone and install locally:
git clone https://github.com/Kirankumarvel/DimenSight.git
cd DimenSight
pip install -r requirements.txt
```

### Basic Usage
```python
from dimensight import Visualizer
import pandas as pd

# Load your data
df = pd.read_csv('data/features.csv')
X = df.drop('label', axis=1)
y = df['label']

# Create and compare projections
viz = Visualizer()
viz.fit_transform(X, y)
viz.plot_comparison()
```

## 📂 Data Preparation

### Supported Input Formats
- CSV/Excel files
- NumPy arrays
- Pandas DataFrames
- scikit-learn Bunch objects

### Sample Data Structure
```csv
feature1,feature2,...,featureN,label
0.12,0.45,...,0.87,class1
0.34,0.21,...,0.65,class2
```

## 🛠️ Advanced Configuration

### Algorithm Parameters
```python
# Customize reduction parameters
viz = Visualizer(
    pca_params={'n_components': 3},
    tsne_params={'perplexity': 40},
    umap_params={'n_neighbors': 15}
)
```

### Visualization Options
```python
viz.plot(
    dimensions=3,               # 2 or 3 dimensions
    colormap='viridis',         # Matplotlib colormap
    point_size=30,              # Marker size
    alpha=0.7,                 # Point transparency
    save_path='output.png'     # Save location
)
```

## 📊 Method Comparison

| Algorithm | Strengths | Best For | Runtime |
|-----------|-----------|----------|---------|
| PCA | Preserves variance | Linear structures | ⚡ Fastest |
| t-SNE | Reveals local clusters | Non-linear manifolds | 🐢 Slow |
| UMAP | Balances local/global | Large datasets | ⏱️ Medium |

## 💡 Pro Tips

1. **For Large Datasets**:
   ```python
   # Use UMAP for better scaling
   viz = Visualizer(methods=['umap'])
   ```

2. **Interactive Exploration**:
   ```python
   # Enable Plotly interactive mode
   viz.plot(interactive=True)  # Requires plotly
   ```

3. **Feature Importance**:
   ```python
   # Get PCA component loadings
   loadings = viz.pca.components_
   ```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory Error | Use `TSNE(angle=0.8)` or subsample data |
| Strange Clusters | Adjust UMAP's `n_neighbors` or t-SNE's `perplexity` |
| Points Overlapping | Decrease point size or increase alpha |

## 🤝 Contributing

We welcome contributions in:
- New algorithms (LLE, Isomap)
- Interactive visualization
- GPU acceleration
- Automated parameter tuning

**Contribution Guide**:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

MIT License - Free for academic and commercial use

---

**Research Tip**: Combine with [scikit-learn](https://scikit-learn.org/) for end-to-end machine learning workflows! 🚀

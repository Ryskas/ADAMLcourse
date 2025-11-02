import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from minisom import MiniSom
from scipy.io import arff

# 1. Linear vs Nonlinear DR on Bike Sharing Rental
def load_arff_dataset(path):
	data, meta = arff.loadarff(path)
	import pandas as pd
	df = pd.DataFrame(data)
	for col in df.select_dtypes([object]).columns:
		df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
	return df, meta.names()


# Load Bike Sharing dataset
df_bike, attr_bike = load_arff_dataset('dataset.arff')
cat_cols = ['season', 'holiday', 'workingday', 'weather']
df_bike_enc = pd.get_dummies(df_bike, columns=cat_cols)
X_bike_features = df_bike_enc.drop(['count'], axis=1).values.astype(float)
y_bike = df_bike_enc['count'].values.astype(float)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_bike_features)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_bike_features)

# Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_bike, palette='viridis', legend=False)
plt.title('PCA of Bike Sharing')
plt.subplot(1,2,2)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_bike, palette='viridis', legend=False)
plt.title('t-SNE of Bike Sharing')
plt.tight_layout()
plt.savefig('bike_sharing_dr_comparison.png')

# Prediction models
def evaluate_model(X, y, name):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	rf = RandomForestRegressor(random_state=42)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	print(f'{name} - Random Forest: MSE={mse:.2f}, MAE={mae:.2f}')
	
	mlp = MLPRegressor(random_state=42, max_iter=1000)
	mlp.fit(X_train, y_train)
	y_pred_mlp = mlp.predict(X_test)
	mse_mlp = mean_squared_error(y_test, y_pred_mlp)
	mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
	print(f'{name} - MLP: MSE={mse_mlp:.2f}, MAE={mae_mlp:.2f}')

print('Model performance on original features:')
evaluate_model(X_bike_features, y_bike, 'Original')
print('Model performance on PCA features:')
evaluate_model(X_pca, y_bike, 'PCA')
print('Model performance on t-SNE features:')
evaluate_model(X_tsne, y_bike, 't-SNE')

# 2. Visualize MNIST-784 with SOM
X_mnist, attr_mnist = load_arff_dataset('mnist_784.arff')
X_mnist = np.array(X_mnist, dtype=float)
X_mnist_features = X_mnist[:, :-1]
y_mnist = X_mnist[:, -1]

# SOM setup
som_size = 20
som = MiniSom(som_size, som_size, X_mnist_features.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som.train_random(X_mnist_features, 1000)

# Visualize SOM
plt.figure(figsize=(8,8))
for i, x in enumerate(X_mnist_features):
	w = som.winner(x)
	plt.text(w[0]+0.5, w[1]+0.5, str(int(y_mnist[i])), color=plt.cm.tab10(int(y_mnist[i])%10), fontdict={'weight': 'bold', 'size': 7})
plt.xlim([0, som_size])
plt.ylim([0, som_size])
plt.title('SOM visualization of MNIST-784')
plt.savefig('mnist_som_visualization.png')

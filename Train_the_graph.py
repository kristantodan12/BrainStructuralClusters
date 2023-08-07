### This script aims to identify the brain structural networks/clusters using graph convolutional networks coupled with deep graph infomax (https://arxiv.org/pdf/1809.10341.pdf)
### The script is adapted from https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html with modifications

# Import libraries
from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph.layer import GCN, DeepGraphInfomax
from stellargraph.utils import plot_history
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
import scipy.io as sio
from stellargraph import StellarGraph

# Load data
node_properties = sio.loadmat('area_properties_a.mat') #import the local properties of each node/brain area (360 brain areas: thickness, myelination, curvature, sulcus depth)
node_properties = node_properties['area_properties_a']
edges = sio.loadmat('edges_10a.mat') #import connection between node/brain area (binary matrix, 360x360, with the 10% strongest connections are coded '1', others '0'
edges = edges['edges_10a']
edges1 = (edges[:, 0])
edges2 = (edges[:, 1])


# Create the graph
square_numeric_edges = pd.DataFrame(
    {"source": edges1, "target": edges2}
)
G = StellarGraph(node_properties, square_numeric_edges, is_directed=False) #Create graph for stellargraph


# Data Generators
fullbatch_generator = FullBatchNodeGenerator(G, sparse=False) #generate original data
gcn_model = GCN(layer_sizes=[64, 64], activations=["relu","relu"], generator=fullbatch_generator) #creat GCN model

corrupted_generator = CorruptedGenerator(fullbatch_generator)  #generate corrupted data
gen = corrupted_generator.flow(G.nodes()) #corrupted data

# Model creation and training
infomax = DeepGraphInfomax(gcn_model, corrupted_generator) #Here we have the GCN and corrupted generator
x_in, x_out = infomax.in_out_tensors() #function to create input and output for unsupervised, input:graph, output:0,1

model = Model(inputs=x_in, outputs=x_out)
opt = Adam(learning_rate=0.001)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=opt)
epochs = 300
es = EarlyStopping(monitor="loss", min_delta=0, patience=50)
history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
plot_history(history)

# Extract embedding vectors
x_emb_in, x_emb_out = gcn_model.in_out_tensors() #function to create input and output for unsupervised

# for full batch models, squeeze out the batch dim (which is 1)
x_out = tf.squeeze(x_emb_out, axis=0)
emb_model = Model(inputs=x_emb_in, outputs=x_out)

# Reduction of the embeddings using T-SNE
all_embeddings = emb_model.predict(fullbatch_generator.flow(G.nodes()))
trans = TSNE(n_components=2).fit_transform(all_embeddings)
emb_transformed = pd.DataFrame(trans.fit_transform(all_embeddings), index=G.nodes())
tsne_df_scale = pd.DataFrame(trans, columns=['tsne1', 'tsne2'])

# Clustering of the features from T-SNE using k-means clustering
kmeans_tsne_scale = KMeans(n_clusters=7, n_init=10, max_iter=400, init='k-means++', random_state=20).fit(tsne_df_scale)
labels_tsne_scale = kmeans_tsne_scale.labels_
clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)

# Save the clustering result
clusters_tsne_scale.to_csv(r'cluster.csv', index=False)




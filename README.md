# adme_tox
Python script that implements a random forest algorithm to predict several ADME/Tox classifications of bioactive molecules accompanied with a visualization technique called Uniform Manifold Approximation Projection (UMAP). This work is an amalgamation of a great many previous work by fellow researchers [ref] with an extension towards our own research work on predicting ion fragmentation by a mass spectrometer (MS).

The script is step-by-step implementation of our approach and is meant for sharing, studying, and critiquing by fellow researchers who are new and interested in the topic. I also include references when appropriate. The script follows the workflow below:

(image)

## Dependencies

```ruby
python: 3.6.9
numpy: 1.18.1
pandas: 0.25.1
sklearn: 0.22.1          
scipy: 1.4.1 
matplotlib: 2.2.3 
rdkit: 2018.09.2.0
tensorflow: 1.14.0
keras: 2.0.9
gpyopt: 1.2.5          
seaborn: 0.9.0
imageio: 2.8.0
umap: 0.3.10
```

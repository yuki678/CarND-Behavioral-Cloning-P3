from IPython.display import display, Image, SVG
from keras.models import load_model, Model
from functools import reduce
import graphviz as gv
import functools
import json
import matplotlib.pyplot as plt

def loadModel(modelPath):
    """
    Loads the model `modelPath`.
    """
    model = load_model(modelPath)
    return model

def getLayerConfig(layer):
    """
    Extract configuration from `layer`.
    """
    layerType = layer.__class__.__name__
    output = { 'type': layerType }
    config = layer.get_config()
    if layerType == 'Lambda':
        _, x, y, d = config['batch_input_shape']
        output['input'] = (x, y, d)
    if layerType == 'Cropping2D':
        output['cropping'] = config['cropping']
    if layerType == 'Conv2D':
        output['activation'] = config['activation']
        output['strides'] = config['strides']
        output['filters'] = config['filters']
        output['kernel'] = config['kernel_size']
    if layerType == 'Dense':
        output['activation'] = config['activation']
        output['units'] = config['units']
    return output

def compressLayers(layers):
    """
    Compress the common layers into a single structure for visualization.
    """
    def reductor(acc, layer):
        if len(acc) == 0:
            acc.append(layer)
            return acc
        
        last = acc[-1]
        if last['type'] == layer['type']:
            try:
                last['items'].append(layer)
            except KeyError:
                acc[-1] = { 'type': layer['type'], 'items': [last, layer]}
        else:
            acc.append(layer)
        return acc
    
    return reduce(reductor, layers, [])

def createNode(nodeName, layer, g, style='filled', fillcolor='white', fontcolor='black'):
    """
    Creates a node with the information from `layer` in the `g`.
    """
    type = layer['type']
    label = type + '\n'
    for key, value in layer.items():
        if (key != 'type'):
            label += '{}: {}\n'.format(key, value)
    g.node(nodeName, label=label, style=style, fillcolor=fillcolor, fontcolor=fontcolor)

def visualizeLayers(layers, outputPath):
    """
    Visualize `layers` and store the image at `outputPath`.
    """
    fillcolors = { 'Conv2D':'#AAAAAA', 'Dense':'#006699' }
    g = gv.Digraph(format='png')
    for index, layer in enumerate(layers):
        nodeName = str(index)
        try:
            items = layer['items']
            subGraphType = layer['type']
            fillcolor = fillcolors[subGraphType]
            g.node(nodeName, label=subGraphType, style='filled', fillcolor=fillcolor, fontcolor='white')
            subG = gv.Digraph(format='png')
            for i, subLayer in enumerate(items):
                subNodeName = nodeName + str(i)
                createNode(subNodeName, subLayer, subG, fillcolor=fillcolor, fontcolor='white')
                if i != 0:
                    subG.edge(nodeName + str(i - 1), subNodeName)
                    
            g.subgraph(subG)
            
        except KeyError:
            createNode(nodeName, layer, g)
        if index != 0:
            g.edge(str(index - 1), nodeName)
            
    styles = {
        'graph': {
        },
        'nodes': {
            'fontname': 'Helvetica',
            'shape': 'rectangle'
        },
        'edges': {
            'arrowhead': 'open'
        }
    }
    g.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    g.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    g.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    
    g.render(outputPath)
    figure = Image(outputPath + '.png')
    display(figure)
    
def visualizeModel(modelPath, imagePath):
    """
    Visualize the model found at `modelPath` to a SVG at `imagePath`
    """
    model = loadModel(modelPath)
    layerData = list(map(getLayerConfig, model.layers))
    compressedLayers = compressLayers(layerData)
    visualizeLayers(compressedLayers, imagePath)
    
model_path = 'models/model.h5'
loadModel(model_path).summary()
visualizeModel(model_path, 'images/model')

# dict_keys(['loss', 'val_loss'])
loss = [0.02814892864289296, 0.004734130184661697, 0.002493900195834608]
valid_loss = [0.027936478145135307, 0.020000185329467818, 0.02095587708179046]

### plot the training and validation loss for each epoch
plt.figure(figsize=(10,7))
plt.plot(loss)
plt.plot(valid_loss)
plt.title('Model Loss (MSE)')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.show()
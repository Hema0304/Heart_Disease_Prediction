from graphviz import Digraph

dot = Digraph(format='png')
dot.attr(rankdir='TB', size='8,10')
dot.attr('node', shape='box', style='filled', fontname="Helvetica")

dot.node('start', 'Start', shape='ellipse', fillcolor='#99ccff')  # Light Blue
dot.node('load', 'Load & Preprocess Dataset\n(StandardScaler)', fillcolor='#b3d9ff')  # Sky Blue
dot.node('init', 'Initialize ACO Parameters\n(ants, iterations, pheromones)', fillcolor='#80bfff')  # Soft Blue
dot.node('loop_subsets', 'For Each Subset (n=6)', fillcolor='#66b3ff')  # Medium Blue
dot.node('pheromone', 'Initialize Pheromone Vector', fillcolor='#4da6ff')  # Bright Blue
dot.node('iteration', 'For Each Iteration (n=100)', fillcolor='#3399ff')  # Clear Blue
dot.node('generate', 'Generate Subsets\n(Based on Pheromone Prob)', fillcolor='#1a8cff')  # Vivid Blue
dot.node('evaluate', 'Evaluate Subsets (F1 Score)\nUsing 3 Models & Stratified 5-Fold CV', fillcolor='#0073e6')  # Deep Sky Blue
dot.node('update', 'Update Pheromone Matrix', fillcolor='#0059b3')  # Dark Blue
dot.node('best_subset', 'Select Best Subset\n(Post-filter with Mutual Info)', fillcolor='#004080')  # Navy Blue
dot.node('save', 'Save Selected Subsets\nto File', fillcolor='#00264d')  # Very Dark Blue
dot.node('end', 'End', shape='ellipse', fillcolor='#99ccff')  # Light Blue (same as start)

dot.edge('start', 'load')
dot.edge('load', 'init')
dot.edge('init', 'loop_subsets')
dot.edge('loop_subsets', 'pheromone')
dot.edge('pheromone', 'iteration')
dot.edge('iteration', 'generate')
dot.edge('generate', 'evaluate')
dot.edge('evaluate', 'update')
dot.edge('update', 'iteration', label='Repeat until iterations done')
dot.edge('iteration', 'best_subset', label='After all iterations')
dot.edge('best_subset', 'loop_subsets', label='Next subset')
dot.edge('loop_subsets', 'save', label='After all subsets')
dot.edge('save', 'end')

dot.render('aco_architecture_diagram_blue', view=True)

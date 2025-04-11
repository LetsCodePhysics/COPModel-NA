# Import libraries.
import warnings
warnings.filterwarnings('ignore')
import datetime
import networkx as nx
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.cluster.hierarchy import dendrogram
from itertools import chain, combinations
from scipy.cluster import hierarchy
from copy import deepcopy
# from wordcloud import WordCloud

# Read in the spreadsheet and count the number of drawings.
# If you use a different Google Sheet, replace the identifier between /d/ and
# /export with the sheet's identifier. Then re-run this cell.
# Make sure sharing is turned on for anyone with the link to view.
# The tab you want must be the first in the sheet. There's a setting to find a sheet
# by name but it's been giving me an error...
# full_database = pd.read_excel("https://docs.google.com/spreadsheets/d/105X8gfRuPtTH79czJQYMcMD1zPjopOWvGRgGOiSFjnc/export")#,sheet_name=sheet_name)
# full_database = pd.read_excel("https://docs.google.com/spreadsheets/d/1TGk3QrpYbnSmVDT-c52PDL_ysp115mEm_mloAbwPRIs/export",engine='openpyxl')#,sheet_name=sheet_name)


# Define a function that creates the desired network.
def MakeGraph(drawings_in,full_database,node_selection='min_weight',min_node_weight=1,nodes_in=[],N_random=None):
  # INPUTS:
  # drawings = ['Drawing 1', 'Drawing 3', etc. indicating drawings to include in the network]
  # full_database = the full set of data read in from Google Sheets
  # node_selection can equal
     # 'min_weight' - The graph will include only nodes of weight >= min_node_weight
     # 'nodes_list' - The graph will include only nodes in the list nodes_in
  # OUTPUTS:
  # function returns the graph (network) G based on the included drawings
  # function also creates network diagram color-coded by category

  # Set up empty dataframe.
  df = pd.DataFrame()

  # Set up columns of elements and categories.
  df['Element'] = full_database['Element']
  df['Category'] = full_database['Category']

  # Set up frequency column. We'll calculate it in the next cell.
  df['Frequency'] = np.zeros(len(df['Element']), dtype=int)

  # Set up element number column. This is to serve as a label.
  df['Element Number'] = np.zeros(len(df['Element']), dtype=int)
  df['Element Key'] = np.empty(len(df['Element']), dtype=str)

  # If node_selection is nodes_list, set min_node_weight higher than can be reached.
  if node_selection == 'nodes_list':
    min_node_weight = len(df['Element']) + 100

  # Set up the coincidence table for elements.
  # In this table, c_ij = number of drawings elements i and j occur in together.
  # Also count number of Demographic items to help offset loops later.
  n_Demographics = 0
  n_Elements = 0
  AddKeyNumber = False

  # Create coincidence table of zeros.
  for j in range(len(full_database['Element'])):
    category = full_database['Category'][j]
    element  = full_database['Element'][j]
    if category != 'Demographic':
      df[element] = np.zeros(len(full_database['Element']))
      n_Elements += 1
      df.loc[j,'Element Number'] = n_Elements
      if AddKeyNumber:
        df.loc[j,'Element Key'] = str(n_Elements) + element
      else:
        df.loc[j,'Element Key'] = element
    else:
      n_Demographics += 1
      df.loc[j,'Element Key'] = ''
  
  drawings_to_use = []
  if (N_random != None):
    # Choose N drawings randomly.
    drawings_to_use = np.random.choice(drawings_in,N_random)
  else:
    drawings_to_use = drawings_in
  # Create coincidence table from the dataset.
  # Count the frequency and coincidences of elements.
  for i in range(n_Demographics,len(df['Element'])):
    frequency = 0
    for sn in drawings_to_use:
      frequency += full_database[sn][i]=='y'
    # print(df['Frequency'][i], frequency,len(df['Frequency']),i)
    df.loc[i,'Frequency'] = frequency
    for j in range(i+1,len(df['Element'])):
      count = 0
      for sn in drawings_to_use:
        count += full_database[sn][i]=='y' and full_database[sn][j]=='y'
      df.loc[j,df['Element'][i]] = count
  
  # Create the graph.
  G = nx.Graph()
  G.clear()
  
  # Identify maximum frequency, for scaling the diagram.
  max_frequency = max(df['Frequency'][1:len(df['Frequency'])])
  # Pull off the non-zero frequency values to determine the minimum frequency,
  # for scaling the diagram.
  NonZeroFrequencies = []
#   for a in df['Frequency'][1:len(df['Frequency'])]:
#     if a >= min_node_weight: NonZeroFrequencies.append(a)
  for i in range(1,len(df['Frequency'])):
    if node_selection == 'nodes_list':
        if df['Element'][i] in nodes_in and df['Frequency'][i]>0: NonZeroFrequencies.append(df['Frequency'][i])
    if node_selection == 'min_weight':
        if df['Frequency'][i]>min_node_weight: NonZeroFrequencies.append(df['Frequency'][i])
  min_frequency = min(NonZeroFrequencies)
  # Create node size scale and edge size scale.
  node_scale = max_frequency/min_frequency
  edge_scale = 0
  
  # Create the edges. This automatically creates the nodes.
  for i in range(n_Demographics,len(df['Element'])):
    if df['Frequency'][i] >= min_node_weight or df['Element'][i] in nodes_in:
      ei = df['Element'][i]
      for j in range(i+1,len(df['Element'])):
        if df['Frequency'][j] >= min_node_weight or df['Element'][j] in nodes_in:
          ej = df['Element'][j]
          if not math.isnan(df[ei][j]) and df[ei][j] > 0 and df['Category'][j] != 'Demographic':
            G.add_edge(ei,ej,weight=df[ei][j],color='b')
            edge_scale = max(edge_scale,df[ei][j])
  # Color-code nodes based on category.
  for j in range(n_Demographics,len(df['Element'])):
    category = df['Category'][j]
    element  = df['Element'][j]
    # Ignore Demographic rows.
    ThisIsAnElement = ((category == 'Practice') or (category == 'Goal') or (category == 'Member'))
    if ThisIsAnElement:
      # Ignore elements with insignificant frequency. For example, if you are using a disaggregated subgroup
      # and none of them included a given element.
      ThisIsAnElement = df['Frequency'][j] >= min_node_weight or df['Element'][j] in nodes_in
    if ThisIsAnElement and element in G.nodes():
      if category == 'Practice':
        G.nodes.data()[element]['color']='r'
      elif category == 'Goal':
        G.nodes.data()[element]['color']='b'
      elif category == 'Member':
        G.nodes.data()[element]['color']='g'
      # Set default node border.
      G.nodes.data()[element]['edgecolor']=G.nodes.data()[element]['color']
      G.nodes.data()[element]['linewidth']=1.0
      G.nodes.data()[element]['weight']=df['Frequency'][j]
      G.nodes.data()[element]['category']=category
  G.edge_scale = edge_scale
  G.node_scale = node_scale
  G.n_Demographics = n_Demographics
  # Create the network diagram. Note that repeating the pos = line will rearrange the nodes.
  # Comment out this line to keep the same arrangement but change cosmetics.
  pos = nx.spring_layout(G)
  edges = G.edges()
  nodes = G.nodes()
  # Set properties of nodes and edges.
  weights = [G[u][v]['weight']/G.edge_scale for u,v in edges] # Size of edges.
#   print(nodes.data())
  ncolors = [G.nodes.data()[u]['color'] for u in nodes] # Color of nodes.
  ecolors = [G.nodes.data()[u]['edgecolor'] for u in nodes] # Colors of node borders.
  lwidths = [G.nodes.data()[u]['linewidth'] for u in nodes] # Width of node borders.
  sizes = [G.nodes.data()[u]['weight']*G.node_scale*0.75 for u in nodes] # Size of nodes.
  labeldict = {}
  for j in range(n_Demographics,len(full_database['Element'])):
    # Ignore elements with 0 frequency.
    if df['Frequency'][j] >= min_node_weight or df['Element'][j] in nodes_in:
      labeldict[df['Element'][j]] = str(df['Element Number'][j])
  inv_factor = len(nodes) * 100
  for u,v in edges: # Inverse weight of edges, for centrality distance.
    G[u][v]['weight_inverse'] = int(inv_factor/G[u][v]['weight']) # Convert to int to appease betweenness.

  G.labeldict = labeldict
  G.n_drawings = len(drawings_in)
  G.elements_per_drawing = len(G.nodes) / len(drawings_in)
  G.betweenness = nx.betweenness_centrality(G, weight = 'weight_inverse')
  G.closeness = nx.closeness_centrality(G, distance = 'weight_inverse')
  G.nodestrength = dict(G.degree(weight='weight')) 
  G.normnodestrength = {}
  for key,value in G.nodestrength.items():
    G.normnodestrength[key] = value / ((len(G.nodes)-1)*len(drawings_in))
  G.nodedegree = dict(G.degree)
  G.normnodedegree = {}
  for key,value in G.nodedegree.items():
    G.normnodedegree[key] = value / (len(G.nodes)-1)
  G.nodeweight = {}
  for node in G.nodes:
    G.nodeweight[node] = G.nodes.data()[node]['weight']

  # The draw command.
  # nx.draw(G, pos, with_labels=AddKeyNumber, labels=labeldict, font_size=10, node_color=ncolors, node_size=sizes, linewidths=lwidths, width=weights, edgecolors = ecolors, cmap = 'viridis')
  # Add a legend for the color-coding.
  # plt.text(0.5, 0.95, 'Practice',color='r')
  # plt.text(0.5, 0.87, 'Member',color='g')
  # plt.text(0.5, 0.80, 'Goal',color='b')
  return G

def MakeBootstrapGraph(G,cap_edge_weight=True):
  # Make a bootstrap graph of G by reassigning edge values using a Poisson distribution.
#   G_bootstrap = G
#   for u,v in G.edges:
#     G_bootstrap[u][v]['weight'] = np.random.poisson(lam=G[u][v]['weight'])
  G_bootstrap = nx.Graph()
  G_bootstrap.edge_scale = G.edge_scale
  G_bootstrap.node_scale = G.node_scale
  G_bootstrap.labeldict = G.labeldict
  inv_factor = len(G.nodes()) * 100
  for u,v in G.edges:
    # Randomize edge weight.
    new_weight = np.random.poisson(lam=G[u][v]['weight'])
    if cap_edge_weight:
      # Cap at the node weight of the two nodes u,v.
      new_weight = min(new_weight,G.nodes.data()[u]['weight'],G.nodes.data()[v]['weight'])
    if new_weight > 0:
      G_bootstrap.add_edge(u,v,weight=new_weight)
      G_bootstrap[u][v]['weight_inverse']=int(inv_factor/new_weight) # Convert to int to appease betweenness.
  for node in G:
#     print(G.nodes.data()[node]['weight'])
    if node not in G_bootstrap:
      G_bootstrap.add_node(node)
    G_bootstrap.nodes.data()[node]['weight'] = G.nodes.data()[node]['weight']
    G_bootstrap.nodes.data()[node]['category'] = G.nodes.data()[node]['category']
    G_bootstrap.nodes.data()[node]['color'] = G.nodes.data()[node]['color']
    G_bootstrap.nodes.data()[node]['edgecolor'] = G.nodes.data()[node]['edgecolor']
    G_bootstrap.nodes.data()[node]['linewidth'] = G.nodes.data()[node]['linewidth']
  G_bootstrap.n_drawings = G.n_drawings
  G_bootstrap.betweenness = nx.betweenness_centrality(G_bootstrap, weight = 'weight_inverse')
  G_bootstrap.closeness = nx.closeness_centrality(G_bootstrap, distance = 'weight_inverse')
  G_bootstrap.nodestrength = dict(G_bootstrap.degree(weight='weight')) 
  G_bootstrap.normnodestrength = {}
  for key,value in G_bootstrap.nodestrength.items():
    G_bootstrap.normnodestrength[key] = value / ((len(G_bootstrap.nodes)-1)*G_bootstrap.n_drawings)
  G_bootstrap.nodedegree = dict(G_bootstrap.degree)
  G_bootstrap.normnodedegree = {}
  for key,value in G_bootstrap.nodedegree.items():
    G_bootstrap.normnodedegree[key] = value / (len(G_bootstrap.nodes)-1)
  G_bootstrap.nodeweight = {}
  for node in G_bootstrap.nodes:
    G_bootstrap.nodeweight[node] = G_bootstrap.nodes.data()[node]['weight']

  return G_bootstrap

def DrawGraph(G,node_size_control=0.75,edge_size_control=1.0,figsize=None,pos=None,font_size=12):
  # Create the network diagram. Note that repeating the pos = line will rearrange the nodes.
  # Comment out this line to keep the same arrangement but change cosmetics.
  if pos == None:
    pos = nx.spring_layout(G)
  else:
    new_pos = {}
    for key,value in pos.items():
      if key in G.nodes:
        new_pos[key] = value
    pos = new_pos
  edges = G.edges()
  nodes = G.nodes()
  # Set properties of nodes and edges.
  weights = [G[u][v]['weight']/G.edge_scale*edge_size_control for u,v in edges] # Size of edges.
  ncolors = [G.nodes.data()[u]['color'] for u in nodes] # Color of nodes.
  ecolors = [G.nodes.data()[u]['edgecolor'] for u in nodes] # Colors of node borders.
  lwidths = [G.nodes.data()[u]['linewidth'] for u in nodes] # Width of node borders.
  sizes = [G.nodes.data()[u]['weight']*G.node_scale*node_size_control for u in nodes] # Size of nodes.

  # The draw command.
  plt.figure(figsize=figsize)
  nx.draw(G, pos, with_labels=False, labels=G.labeldict, font_size=font_size, node_color=ncolors, node_size=sizes, linewidths=lwidths, width=weights, edgecolors = ecolors, cmap = 'viridis')
  # Add a legend for the color-coding.
  plt.text(0.5, 0.95, 'Practice',color='r',fontsize=font_size)
  plt.text(0.5, 0.87, 'Member',color='g',fontsize=font_size)
  plt.text(0.5, 0.80, 'Goal',color='b',fontsize=font_size)

  return pos

# Define a function that checks for the current disaggregation conditions.
# full_database[sn][m] = drawing sn's info for Demographic item in row m
# def DrawingInSubgroup(full_database,sn,demographic,value): # Original version for one demographic item.
def DrawingInSubgroup(full_database,sn,demographics,values):
  # sn = string for drawing number column header, created before calling the function
  if len(demographics) == 0:
    return True
  else:
    answer = True
    for i_demographic in range(len(demographics)):
      demographic = demographics[i_demographic]
      value = values[i_demographic]
      i_check = full_database[full_database['Element']==demographic].index[0]
      answer = answer and full_database[sn][i_check]==value
    return answer

# Create a list of the drawings in a subgroup.
# def MakeSubgroup(full_database,demographic='',value=''): # Originanl version for one demographic item.
def MakeSubgroup(full_database,*argv):
  # Create list of demographics and values to check.
  demographics = []
  values = []
  for i in range(0,len(argv),2):
    demographics.append(argv[i])
    values.append(argv[i+1])
  # Count the number of drawings in the database.
  n_drawings = len(full_database.columns)-2
  # print('n_drawings',n_drawings)
  n_drawings_in = 0
  drawings_in = []
  for n in range(n_drawings):
    sn = 'Drawing '+str(n+1)
    # print(sn,demographic,value)
    if DrawingInSubgroup(full_database,sn,demographics,values):
      # print('here i am')
      n_drawings_in += 1
      drawings_in.append(sn)
    # else:
    #   print('here i aint')
  # print('n_drawings_in',n_drawings_in,drawings_in)
  return drawings_in

def NodeDegreeCosine(G1,G2):
  numerator = 0
  denominator1 = 0
  denominator2 = 0
  for node in G1:
    if G2.has_node(node):
      numerator += G1.degree[node]*G2.degree[node]
      denominator1 += G1.degree[node]**2
      denominator2 += G2.degree[node]**2
    else:
      denominator1 += G1.degree[node]**2
  for node in G2:
    if not G1.has_node(node):
      denominator2 += G2.degree[node]**2
  if denominator1>0 and denominator2>0:
    ndc = numerator / (denominator1*denominator2)**0.5
  else:
    ndc = 0
  return ndc

def NodeWeightCosine(G1,G2):
  numerator = 0
  denominator1 = 0
  denominator2 = 0
  for node in G1:
    if G2.has_node(node):
      numerator += G1.nodes[node]['weight']*G2.nodes[node]['weight']
      denominator1 += G1.nodes[node]['weight']**2
      denominator2 += G2.nodes[node]['weight']**2
    else:
      denominator1 += G1.nodes[node]['weight']**2
  for node in G2:
    if not G1.has_node(node):
      denominator2 += G2.nodes[node]['weight']**2
  if denominator1>0 and denominator2>0:
    nwc = numerator / (denominator1*denominator2)**0.5
  else:
    nwc = 0
  return nwc

def NodeStrengthCosine(G1,G2):
  numerator = 0
  denominator1 = 0
  denominator2 = 0
  for node in G1:
    if G2.has_node(node):
      numerator += NodeStrength(G1,node)*NodeStrength(G2,node)
      denominator1 += NodeStrength(G1,node)**2
      denominator2 += NodeStrength(G2,node)**2
    else:
      denominator1 += NodeStrength(G1,node)**2
  for node in G2:
    if not G1.has_node(node):
      denominator2 += NodeStrength(G2,node)**2
  if denominator1>0 and denominator2>0:
    nsc = numerator / (denominator1*denominator2)**0.5
  else:
    nsc = 0
  return nsc

def CategoryNodeDegreeCosine(G1,G2,nodes):
  numerator = 0
  denominator1 = 0
  denominator2 = 0
  for node in nodes:
    if G1.has_node(node) and G2.has_node(node):
      numerator += G1.degree[node]*G2.degree[node]
      denominator1 += G1.degree[node]**2
      denominator2 += G2.degree[node]**2
    elif G1.has_node(node):
      denominator1 += G1.degree[node]**2
    elif G2.has_node(node):
      denominator2 += G2.degree[node]**2
  if denominator1>0 and denominator2>0:
    ndc = numerator / (denominator1*denominator2)**0.5
  else:
    ndc = 0
  return ndc

def CategoryNodeWeightCosine(G1,G2,nodes):
  numerator = 0
  denominator1 = 0
  denominator2 = 0
  for node in nodes:
    if G1.has_node(node) and G2.has_node(node):
      numerator += G1.nodes[node]['weight']*G2.nodes[node]['weight']
      denominator1 += G1.nodes[node]['weight']**2
      denominator2 += G2.nodes[node]['weight']**2
    elif G1.has_node(node):
      denominator1 += G1.nodes[node]['weight']**2
    elif G2.has_node(node):
      denominator2 += G2.nodes[node]['weight']**2
  if denominator1>0 and denominator2>0:
    nwc = numerator / (denominator1*denominator2)**0.5
  else:
    nwc = 0
  return nwc

def CategoryNodeStrengthCosine(G1,G2,nodes):
  numerator = 0
  denominator1 = 0
  denominator2 = 0
  for node in nodes:
    if G1.has_node(node) and G2.has_node(node):
      numerator += NodeStrength(G1,node)*NodeStrength(G2,node)
      denominator1 += NodeStrength(G1,node)**2
      denominator2 += NodeStrength(G2,node)**2
    elif G1.has_node(node):
      denominator1 += NodeStrength(G1,node)**2
    elif G2.has_node(node):
      denominator2 += NodeStrength(G2,node)**2
  if denominator1>0 and denominator2>0:
    nsc = numerator / (denominator1*denominator2)**0.5
  else:
    nsc = 0
  return nsc

def EEJ(G1,G2):
  numerator = 0
  denominator = 0
  for u,v in G1.edges:
    if G2.has_edge(u,v):
      numerator += 1
    else:
      denominator += 1
  for u,v in G2.edges:
    denominator += 1
    
  return numerator / denominator

# Functions to get backbone network.

# Fractional Edge Weight: How weighty is this edge compared to all other edges coming out of node_center?
def FractionalEdgeWeight(G,node_center,node_neighbor):
  denominator = 0
  for node in G.nodes():
    if (node_center,node) in G.edges:
      denominator += G.edges[node_center,node]['weight']
  return G.edges[node_center,node_neighbor]['weight'] / denominator

# Fhat: How many edges is this edge weightier than coming out of node_center?
def Fhat(G,node_center,node_neighbor):
  numerator = 0
  for node in G.nodes():
    if (node_center,node) in G.edges:
      numerator += FractionalEdgeWeight(G,node_center,node) <= FractionalEdgeWeight(G,node_center,node_neighbor)
  return numerator / G.degree[node_center]

# Is this edge significant to either node1 or node2?
def IsEdgeSignificant(G,node1,node2,alpha):
  return 1-Fhat(G,node1,node2)<alpha or 1-Fhat(G,node2,node1)<alpha

# Create the backbone network of G using significance alpha.
# alpha = 0 is less inclusive.
# alpha = 1 is more inclusive.
# NB: This takes a LONG time to run.
def Backbone(G,alpha):
  G_backbone = G
  for u,v in G_backbone.edges:
    if not IsEdgeSignificant(G_backbone,u,v,alpha):
      G_backbone.remove_edge(u, v)
  return G_backbone

# Strength of node.
def NodeStrength(G,node):
  strength = 0
  for node2 in G:
    if (node,node2) in G.edges and node != node2:
      strength += G.edges[node,node2]['weight']
  return strength

# Strength of all nodes. This is like degree, except it adds edge weights together.
def AllStrength(G):
  all_strength = 0
  for node in G:
    all_strength += NodeStrength(G,node)
  return all_strength

# Modularity = how well the network is separated into smaller clusters.
def Modularity(G,clusters):
  Q = 0
  m = 0.5 * AllStrength(G)
  for cluster in clusters:
    for node1 in cluster:
      for node2 in cluster:
        if (node1,node2) in G.edges:
          Q += G.edges[node1,node2]['weight']
        Q -= NodeStrength(G,node1)*NodeStrength(G,node2)/(2*m)
  Q = Q / (2*m)
  return Q

# Purity tests.

# Count the number of nodes that cluster1 and cluster2 have in common.
# This is n_ij in https://www.sciencedirect.com/science/article/pii/S0378873321000307?casa_token=Y4KM5fBX43MAAAAA:ivqcUlFsXD2HZwQ5_QMvKApMMx85F0kiFqWw5PQCYANsXKqLh0_AzIx1BHbm3KS0m_Z5UpDNvc4
def CommonCount(cluster1,cluster2):
  count = 0
  for node in cluster1:
    count += node in cluster2
  return count

# Calculate the purity of cluster wrt a set of other_clusters.
# cluster = a single cluster, formatted as a list of node names
# other_clusters = a set of lists of node names
def Purity(cluster,other_clusters):
  purity = max(CommonCount(cluster,other_cluster) for other_cluster in other_clusters) / len(cluster)
  return purity

# Identify the other_cluster in other_clusters that contains the maximum number of nodes in cluster.
def ClusterOfMaxOverlap(cluster,other_clusters):
  maximum_overlap = 0
  cluster_of_max_overlap = []
  for other_cluster in other_clusters:
    count_overlap = 0
    for node in cluster:
      count_overlap = CommonCount(cluster,other_cluster)
    if count_overlap > maximum_overlap:
      cluster_of_max_overlap = other_cluster
      maximum_overlap = count_overlap
  return cluster_of_max_overlap

# Calculate the purity of a set of clusters wrt a set of other_clusters.
# Each argument is a forzenset of list of node names
def PurityOfClustering(clusters,other_clusters):
  purity1 = sum(Purity(cluster,other_clusters)*len(cluster) for cluster in clusters) / sum(len(cluster) for cluster in clusters)
  purity2 = sum(Purity(other_cluster,clusters)*len(other_cluster) for other_cluster in other_clusters) / sum(len(other_cluster) for other_cluster in other_clusters)
  return 2*purity1*purity2 / (purity1+purity2)

# Calculate the F-Measure of cluster wrt a set of other_clusters.
def FMeasureCluster(cluster,other_clusters):
  # Find the cluster of maximum overlap from other_clusters.
  cluster_of_max_overlap = ClusterOfMaxOverlap(cluster,other_clusters)
  n = CommonCount(cluster,cluster_of_max_overlap)
  m = len(cluster_of_max_overlap)
  return 2*n / (len(cluster) + m)

# https://www.sciencedirect.com/science/article/pii/S0378873321000307?casa_token=Y4KM5fBX43MAAAAA:ivqcUlFsXD2HZwQ5_QMvKApMMx85F0kiFqWw5PQCYANsXKqLh0_AzIx1BHbm3KS0m_Z5UpDNvc4#:~:text=2.3.2.-,F%2DMeasure,-The%20F%2DMeasure
def FMeasure(clusters,other_clusters):
  F1 = sum(FMeasureCluster(cluster,other_clusters) for cluster in clusters) / len(clusters)
  F2 = sum(FMeasureCluster(other_cluster,clusters) for other_cluster in other_clusters) / len(other_clusters)
  return 2*F1*F2 / (F1+F2)
  return F

def most_central_edge(G):
  centrality = nx.edge_betweenness_centrality(G, weight="weight")
  return max(centrality, key=centrality.get)

def DrawDendrogram(G,N_nodes=None,big_nodes=None):
    communities = list(nx.community.girvan_newman(G,most_valuable_edge=most_central_edge)) # List of lists of sets. communities[n] = a list of sets partitioning G at level n 
    if big_nodes == None:
      if N_nodes == None:
        N_nodes = len(G.nodes)
      big_nodes = sorted(G.nodes, key=lambda x: G.nodes[x]['weight'], reverse=True)[0:N_nodes]
    # Count the number of levels required for each node to be split off into its own cluster.
    levels = []
    for node in big_nodes:
#         print(node)
        iteration = 0
        level = 0
        while level == 0:
            for cluster in communities[iteration]:
                if node in cluster and len(cluster) == 1:
                    level = iteration
#                     print(node,level)
            iteration += 1
        levels.append(level)
    ydist = []
    for n1 in range(len(big_nodes)):
        for n2 in range(n1+1,len(big_nodes)):
            ydist.append(abs(levels[n2]-levels[n1]))
#     print(len(ydist))

    Z = hierarchy.linkage(ydist, 'single')
    for row in Z: # Shift tree for visibility at end.
      row[2] += 2
    dng = hierarchy.dendrogram(Z,labels=big_nodes,orientation='left')
    return levels,Z,big_nodes,dng

def DetectClusters(G,weight='weight',method='fast-greedy'):
  # Identify clusters in the network using the specified method.
  if method == 'louvain_communities':
    return nx.community.louvain_communities(G,weight=weight)
  if method == 'label_propagation_communities':
    return nx.community.label_propagation_communities(G)
  if method == 'asyn_lpa_communities':
    return nx.community.asyn_lpa_communities(G,weight=weight)
  if method == 'naive_greedy_modularity_communities':
    return nx.community.naive_greedy_modularity_communities(G,weight=weight)
  if method == 'kernighan_lin_bisection':
    return nx.community.kernighan_lin_bisection(G,weight=weight)
  if method == 'fast-greedy' or method == 'greedy_modularity_communities':
    return nx.community.greedy_modularity_communities(G,weight=weight)
  # Set fast-greedy as default.
  return nx.community.greedy_modularity_communities(G,weight=weight)

# Run a series of N bootstrap clustering tests and average comparison measures.
def BootStrapTest(drawings_in,full_database,N,threshold=0.50,print_output=True,file_out='bootstrap_output.txt',clustering_method='fast-greedy'):
  # drawings_in = list of drawings to include in subset
  # full_database = full database of drawing elements
  # N = number of bootstrap datasets to test

  # Returns a dictionary where each element in the database is a key and each value is the number
  # of times that element ended up in its original cluster.
  # Also calculate the average and standard deviation of the NDC, EEJ, and purity between
  # the network and the N bootstraps.

  NDC_list    = []
  NWC_list    = []
  NSC_list    = []
  EEJ_list    = []
  purity_list = []
  F_list      = []

  # Create graph of original dataset.
  G_original = MakeGraph(drawings_in,full_database)

  if print_output:
    print('original graph',G_original,AllStrength(G_original))
  # Create clusters for original dataset.
  original_clusters = DetectClusters(G_original, weight='weight', method=clustering_method)

  # Create dictionary to count fraction of times a node ends up in its original cluster.
  original_frequency = {}
  for node in G_original:
    original_frequency[node] = 0
    original_frequency[node+' probabilities'] = [] # Create list of probabilities for convergence test, to be graphed versus iteration number.

  # Create N bootstraps and test for how many times each element ends up in their original cluster.
  time_start = datetime.datetime.now()
  for n in range(N):
    G_bootstrap = MakeBootstrapGraph(G_original)

    bootstrap_clusters = DetectClusters(G_bootstrap, weight='weight', method=clustering_method)
    for node in G_original.nodes():
      for ib in range(len(bootstrap_clusters)): # ib = number of this cluster in the bootstrap clusters
        if ib < len(original_clusters):
          original_frequency[node] += (node in original_clusters[ib] and node in bootstrap_clusters[ib])
      original_frequency[node+' probabilities'].append(original_frequency[node]/(n+1))

    NDC_list.append(NodeDegreeCosine(G_original,G_bootstrap))
    NWC_list.append(NodeWeightCosine(G_original,G_bootstrap))
    NSC_list.append(NodeStrengthCosine(G_original,G_bootstrap))
    EEJ_list.append(EEJ(G_original,G_bootstrap))
    purity_list.append(PurityOfClustering(original_clusters,bootstrap_clusters))
    F_list.append(FMeasure(original_clusters,bootstrap_clusters))

    time_estimate = (datetime.datetime.now() - time_start).total_seconds() / (n+1) * (N-n) / 3600
    if print_output:
      print('Finished bootstrap graph',n+1,'of',N)
      print('### This run should finish in',time_estimate,'hours. ###')
    
    with open(file_out, 'w') as convert_file: 
      current_time = datetime.datetime.now()
      convert_file.write(str(current_time) + ' ' + str(n+1) + ' of ' + str(N))
      convert_file.write('\n')
      convert_file.write('### This run should finish in '+str(time_estimate)+' hours. ###')
      for key, value in original_frequency.items():
        convert_file.write(key + ' : ' + str(value))
        convert_file.write('\n')


  for node in G_original.nodes():
    original_frequency[node] = original_frequency[node] / N

  NDC_mean    = np.mean(NDC_list)
  NWC_mean    = np.mean(NWC_list)
  NSC_mean    = np.mean(NSC_list)
  EEJ_mean    = np.mean(EEJ_list)
  purity_mean = np.mean(purity_list)
  NDC_std     = np.std(NDC_list)
  NWC_std     = np.std(NWC_list)
  NSC_std     = np.std(NSC_list)
  EEJ_std     = np.std(EEJ_list)
  purity_std  = np.std(purity_list)

  if print_output:
    print('--Bootstrap Test Completed--')
    print('     NDC ',NDC_mean,'±',NDC_std)
    print('     NWC ',NWC_mean,'±',NWC_std)
    print('     NSC ',NSC_mean,'±',NSC_std)
    print('     EEJ ',EEJ_mean,'±',EEJ_std)
    print('  purity ',purity_mean,'±',purity_std)

    print('The following nodes remained in their original clusters more than',threshold*100,'% of the time:')
    output = []
    for node in G_original.nodes():
      if original_frequency[node] > threshold:
        output.append(node)
    print(output)

  original_frequency['NDC']         = NDC_list
  original_frequency['NWC']         = NWC_list
  original_frequency['NSC']         = NSC_list
  original_frequency['EEJ']         = EEJ_list
  original_frequency['purity']      = purity_list
  original_frequency['NDC mean']    = NDC_mean
  original_frequency['NWC mean']    = NWC_mean
  original_frequency['NSC mean']    = NSC_mean
  original_frequency['EEJ mean']    = EEJ_mean
  original_frequency['purity mean'] = purity_mean
  original_frequency['NDC std']     = NDC_std
  original_frequency['NWC std']     = NWC_std
  original_frequency['NSC std']     = NSC_std
  original_frequency['EEJ std']     = EEJ_std
  original_frequency['purity std']  = purity_std
  original_frequency['nodes']       = G_original.nodes()

  current_time = datetime.datetime.now()

  with open(file_out, 'w') as convert_file: 
    convert_file.write('FINISHED at ' + str(current_time))
    convert_file.write('\n')
#     for key, value in original_frequency.items():
#       convert_file.write(key + ' : ' + str(value))
#       convert_file.write('\n')
    output = ''
    for node in G_original.nodes():
      if original_frequency[node] > threshold:
        output+= node + str(', ')
    convert_file.write('--Bootstrap Test Completed--')
    convert_file.write('\n')
    convert_file.write('     NDC '+str(NDC_mean)+' ± '+str(NDC_std))
    convert_file.write('\n')
    convert_file.write('     NWC '+str(NWC_mean)+' ± '+str(NWC_std))
    convert_file.write('\n')
    convert_file.write('     NSC '+str(NSC_mean)+' ± '+str(NSC_std))
    convert_file.write('\n')
    convert_file.write('     EEJ '+str(EEJ_mean)+' ± '+str(EEJ_std))
    convert_file.write('\n')
    convert_file.write('  purity '+str(purity_mean)+' ± '+str(purity_std))
    convert_file.write('\n')
    convert_file.write('The following nodes remained in their original clusters more than'+str(threshold*100)+'% of the time:')
    convert_file.write('\n')
    convert_file.write(output)
    convert_file.write('\n' + 'And now some details...' + '\n')
    for key, value in original_frequency.items():
      convert_file.write(key + ' : ' + str(value))
      convert_file.write('\n')
       
  return original_frequency

def CohensD(mean1,std1,n1,mean2,std2,n2):
  # denominator of Cohen's d is a pooled standard deivation: https://www.statisticshowto.com/pooled-standard-deviation/
  return abs(mean1-mean2) / (np.sqrt(((n1-1)*std1**2+(n2-1)*std2**2)/(n1+n2-2)))

def CohensStar(d):
  # Return indicator of significance for Cohen's d value.
  # if 0.2 <= d < 0.5:
  #   return '^{*}'
  if 0.5 <= d < 0.8:
    return '^{*}'
  if 0.8 <= d:
    return '^{**}'
  return ''

def get_var_name(var):
    # https://www.geeksforgeeks.org/get-variable-name-as-string-in-python/
    for name, value in locals().items():
        if value is var:
            return name

def BootStrapComparison(all_drawings,drawing_subset_1,drawing_subset_2,full_database,N,subset_name_1=None,subset_name_2=None,N_nodes=5,file_out='bootstrapcomparison.txt',time_print=False,centrality_power=2,clustering_method='fast-greedy',min_node_weight=1,num_sig=3,cap_edge_weight=True,measures_in_table=['betweenness']):
  # Carry out N bootstraps on each of the data sets (all_drawings, drawing_subset_1,drawing_subset_2).
  # Calculate the average and standard deviation for NDC, EEJ, and purity between all_drawings and
  # drawing_subset_1, and between all_drawings and drawing_subset_2.
  # Calculate Cohen's d for each of these averages.
  # Print all these values.
  # Return a dictionary with lots of things.

  if subset_name_1==None:
    subset_name_1 = get_var_name(drawing_subset_1)
  if subset_name_2==None:
    subset_name_2 = get_var_name(drawing_subset_2)
  
  NDC_1    = []
  NDC_2    = []
  NWC_1    = []
  NWC_2    = []
  NSC_1    = []
  NSC_2    = []
  EEJ_1    = []
  EEJ_2    = []
  purity_1 = []
  purity_2 = []
  F_1      = []
  F_2      = []

  print('making full network')
  G_full = MakeGraph(all_drawings,full_database,min_node_weight=min_node_weight,node_selection='min_weight')
  clusters_full = DetectClusters(G_full, weight='weight', method=clustering_method)
  print('making network 1')
  G_1_original = MakeGraph(drawing_subset_1,full_database,node_selection='nodes_list',nodes_in=G_full.nodes)
  print('making network 2')
  G_2_original = MakeGraph(drawing_subset_2,full_database,node_selection='nodes_list',nodes_in=G_full.nodes)

#   # Remove insignificant nodes from G_1_original and G_2_original.
#   to_remove = []
#   for node in G_1_original.nodes:
#     if node not in G_full.nodes:
#       to_remove.append(node)
#   for node in to_remove:
#     G_1_original.remove_node(node)
#   to_remove = []
#   for node in G_2_original.nodes:
#     if node not in G_full.nodes:
#       to_remove.append(node)
#   for node in to_remove:
#     G_2_original.remove_node(node)
  
  # Create lists for within-category data.
  categories = ['Goal','Member','Practice']
  category_counts = {'measure':'counts'}
  category_weights = {'measure':'weights'}
  category_strengths = {'measure':'strengths'}
  category_internal_connections = {'measure':'internal connections'}
  category_NDCs = {'measure':'NDCs'}
  category_NWCs = {'measure':'NWCs'}
  category_NSCs = {'measure':'NSCs'}
  category_betweennesses = {'measure':'betweennesses'}
  category_closenesses = {'measure':'closenesses'}
  
  category_lists = {}
  for category in categories:
    category_lists[category + ' full'] = []
    category_lists[category + ' 1'] = []
    category_lists[category + ' 2'] = []
    category_weights[category + ' full'] = 0
    category_weights[category + ' 1'] = 0
    category_weights[category + ' 2'] = 0
    category_counts[category + ' full'] = 0
    category_counts[category + ' 1'] = 0
    category_counts[category + ' 2'] = 0
    category_strengths[category + ' 1'] = []
    category_internal_connections[category + ' 1'] = []
    category_NDCs[category + ' 1'] = []
    category_NWCs[category + ' 1'] = []
    category_NSCs[category + ' 1'] = []
    category_betweennesses[category + ' 1'] = []
    category_closenesses[category + ' 1'] = []
    category_strengths[category + ' 2'] = []
    category_internal_connections[category + ' 2'] = []
    category_NDCs[category + ' 2'] = []
    category_NWCs[category + ' 2'] = []
    category_NSCs[category + ' 2'] = []
    category_betweennesses[category + ' 2'] = []
    category_closenesses[category + ' 2'] = []
    for node in G_full:
      if G_full.nodes.data()[node]['category'] == category:
        category_lists[category + ' full'].append(node)
        category_weights[category + ' full'] += G_full.nodes.data()[node]['weight']
        category_counts[category + ' full'] += 1
    if category_counts[category + ' full'] > 0:
      category_weights[category + ' full'] = category_weights[category + ' full'] / category_counts[category + ' full']
    for node in G_1_original:
      if G_1_original.nodes.data()[node]['category'] == category:
        category_lists[category + ' 1'].append(node)
        category_weights[category + ' 1'] += G_1_original.nodes.data()[node]['weight']
        category_counts[category + ' 1'] += 1
    if category_counts[category + ' 1'] > 0:
      category_weights[category + ' 1'] = category_weights[category + ' 1'] / category_counts[category + ' 1']
    for node in G_2_original:
      if G_2_original.nodes.data()[node]['category'] == category:
        category_lists[category + ' 2'].append(node)
        category_weights[category + ' 2'] += G_2_original.nodes.data()[node]['weight']
        category_counts[category + ' 2'] += 1
    if category_counts[category + ' 2'] > 0:
      category_weights[category + ' 2'] = category_weights[category + ' 2'] / category_counts[category + ' 2']
    
  print('setting up dictionaries')

  big_nodes = sorted(G_full.nodes, key=lambda x: G_full.nodes[x]['weight'], reverse=True)[0:N_nodes]
    
  for node in big_nodes:
    if (node not in G_1_original) or (node not in G_2_original):
      big_nodes.remove(node)
  
  centralities_1 = {}
  centralities_2 = {}
  for node in big_nodes:
    for measure in [' betweenness',' closeness',' normnodedegree',' normnodestrength']:
      key = node + measure
      centralities_1[key] = []
      centralities_2[key] = []

  print('going into bootstrap loop')
  time_start = datetime.datetime.now()
  for n in range(N):
    # print('making bootstraps')
    if time_print: delta_time = datetime.datetime.now()
    G_1 = MakeBootstrapGraph(G_1_original,cap_edge_weight=cap_edge_weight)
    G_2 = MakeBootstrapGraph(G_2_original,cap_edge_weight=cap_edge_weight)
    if time_print: delta_time = datetime.datetime.now() - delta_time
    if time_print: print('bootstrapping',delta_time.total_seconds())
    NDC_1.append(NodeDegreeCosine(G_full,G_1))
    NDC_2.append(NodeDegreeCosine(G_full,G_2))
    NWC_1.append(NodeWeightCosine(G_full,G_1))
    NWC_2.append(NodeWeightCosine(G_full,G_2))
    NSC_1.append(NodeStrengthCosine(G_full,G_1))
    NSC_2.append(NodeStrengthCosine(G_full,G_2))
    EEJ_1.append(EEJ(G_full,G_1))
    EEJ_2.append(EEJ(G_full,G_2))
    # print('clustering')
    clusters_1 = DetectClusters(G_1, weight='weight', method=clustering_method)
    clusters_2 = DetectClusters(G_2, weight='weight', method=clustering_method)
    purity_1.append(PurityOfClustering(clusters_1,clusters_full))
    purity_2.append(PurityOfClustering(clusters_2,clusters_full))
    F_1.append(FMeasure(clusters_1,clusters_full))
    F_2.append(FMeasure(clusters_2,clusters_full))
    # print('getting centrality measures')

#     if time_print: delta_time = datetime.datetime.now()
#     bc1 = nx.betweenness_centrality(G_1, weight = 'weight_inverse')
#     bc2 = nx.betweenness_centrality(G_2, weight = 'weight_inverse')
#     if time_print: delta_time = datetime.datetime.now() - delta_time
#     if time_print: print('betweenness measures',delta_time.total_seconds())

#     if time_print: delta_time = datetime.datetime.now()
#     cc1 = nx.closeness_centrality(G_1, distance = 'weight_inverse')
#     cc2 = nx.closeness_centrality(G_2, distance = 'weight_inverse')
    
#     if time_print: delta_time = datetime.datetime.now() - delta_time
#     if time_print: print('closeness',delta_time.total_seconds())

    bc1 = G_1.betweenness
    bc2 = G_2.betweenness
    cc1 = G_1.closeness
    cc2 = G_2.closeness

    for node in big_nodes:
      key = node + ' betweenness'
      centralities_1[key].append(bc1[node])
      centralities_2[key].append(bc2[node])
      key = node + ' closeness'
      centralities_1[key].append(cc1[node])
      centralities_2[key].append(cc2[node])
      key = node + ' normnodedegree'
      centralities_1[key].append(G_1.normnodedegree[node])
      centralities_2[key].append(G_2.normnodedegree[node])
      key = node + ' normnodestrength'
      centralities_1[key].append(G_1.normnodestrength[node])
      centralities_2[key].append(G_2.normnodestrength[node])
#   category_strengths = {}
#   category_internal_connections = {}
#   category_NDCs = {}
#   category_NWCs = {}
#   category_NSCs = {}
#   category_betweennesses = {}
#   category_closenesses = {}

    if time_print: delta_time = datetime.datetime.now()
    for category in categories:
      category_strengths[category + ' 1'].append(0)
      category_internal_connections[category + ' 1'].append(0)
      category_NDCs[category + ' 1'].append(0)
      category_NWCs[category + ' 1'].append(0)
      category_NSCs[category + ' 1'].append(0)
      category_betweennesses[category + ' 1'].append(0)
      category_closenesses[category + ' 1'].append(0)
      category_strengths[category + ' 2'].append(0)
      category_internal_connections[category + ' 2'].append(0)
      category_NDCs[category + ' 2'].append(0)
      category_NWCs[category + ' 2'].append(0)
      category_NSCs[category + ' 2'].append(0)
      category_betweennesses[category + ' 2'].append(0)
      category_closenesses[category + ' 2'].append(0)

      for node in category_lists[category + ' full']:
        if node in G_1:
          category_strengths[category + ' 1'][n] += NodeStrength(G_1,node)
          for other_node in category_lists[category + ' full']:
            if (node,other_node) in G_1.edges:
              category_internal_connections[category + ' 1'][n] += G_1.edges[node,other_node]['weight']
          category_betweennesses[category + ' 1'][n] += bc1[node]**centrality_power
          category_closenesses[category + ' 1'][n] += cc1[node]**centrality_power
      category_NDCs[category + ' 1'][n] = CategoryNodeDegreeCosine(G_1,G_full,category_lists[category + ' full'])
      category_NWCs[category + ' 1'][n] = CategoryNodeWeightCosine(G_1,G_full,category_lists[category + ' full'])
      category_NSCs[category + ' 1'][n] = CategoryNodeStrengthCosine(G_1,G_full,category_lists[category + ' full'])
      if category_counts[category + ' 1'] > 0:
        category_strengths[category + ' 1'][n] = category_strengths[category + ' 1'][n] / category_counts[category + ' 1']
        category_internal_connections[category + ' 1'][n] = category_internal_connections[category + ' 1'][n] / ((category_counts[category + ' 1']-1)*(category_counts[category + ' 1']-2)*G_1.n_drawings)
        category_betweennesses[category + ' 1'][n] = (category_betweennesses[category + ' 1'][n] / category_counts[category + ' 1'])**(1.0/centrality_power)
        category_closenesses[category + ' 1'][n] = (category_closenesses[category + ' 1'][n] / category_counts[category + ' 1'])**(1.0/centrality_power)
      for node in category_lists[category + ' full']:
        if node in G_2:
          category_strengths[category + ' 2'][n] += NodeStrength(G_2,node)
          for other_node in category_lists[category + ' full']:
            if (node,other_node) in G_2.edges:
              category_internal_connections[category + ' 2'][n] += G_2.edges[node,other_node]['weight']
          category_betweennesses[category + ' 2'][n] += bc2[node]**centrality_power
          category_closenesses[category + ' 2'][n] += cc2[node]**centrality_power
      category_NDCs[category + ' 2'][n] = CategoryNodeDegreeCosine(G_2,G_full,category_lists[category + ' full'])
      category_NWCs[category + ' 2'][n] = CategoryNodeWeightCosine(G_2,G_full,category_lists[category + ' full'])
      category_NSCs[category + ' 2'][n] = CategoryNodeStrengthCosine(G_2,G_full,category_lists[category + ' full'])
      if category_counts[category + ' 2'] > 0:
        category_strengths[category + ' 2'][n] = category_strengths[category + ' 2'][n] / category_counts[category + ' 2']
        category_internal_connections[category + ' 2'][n] = category_internal_connections[category + ' 2'][n] / ((category_counts[category + ' 2']-1)*(category_counts[category + ' 2']-2)*G_2.n_drawings)
        category_betweennesses[category + ' 2'][n] = (category_betweennesses[category + ' 2'][n] / category_counts[category + ' 2'])**(1.0/centrality_power)
        category_closenesses[category + ' 2'][n] = (category_closenesses[category + ' 2'][n] / category_counts[category + ' 2'])**(1.0/centrality_power)
    if time_print: delta_time = datetime.datetime.now() - delta_time
    if time_print: print('category measures',delta_time.total_seconds())

    print('Bootstrap',n+1,'of',N,'completed.')
    time_estimate = (datetime.datetime.now() - time_start).total_seconds() / (n+1) * (N-n) / 3600
    print('### This run should finish in',time_estimate,'hours. ###')

    with open(file_out, 'w') as convert_file: 
      convert_file.write('Bootstrap '+str(n+1)+' of '+str(N)+' completed at ' + str(datetime.datetime.now()))
      convert_file.write('### This run should finish in'+str(time_estimate)+'hours. ###')

  
  NDC_1_mean = np.mean(NDC_1)
  NDC_2_mean = np.mean(NDC_2)
  NWC_1_mean = np.mean(NWC_1)
  NWC_2_mean = np.mean(NWC_2)
  NSC_1_mean = np.mean(NSC_1)
  NSC_2_mean = np.mean(NSC_2)
  EEJ_1_mean = np.mean(EEJ_1)
  EEJ_2_mean = np.mean(EEJ_2)
  purity_1_mean = np.mean(purity_1)
  purity_2_mean = np.mean(purity_2)
  F_1_mean = np.mean(F_1)
  F_2_mean = np.mean(F_2)

  NDC_1_std = np.std(NDC_1)
  NDC_2_std = np.std(NDC_2)
  NWC_1_std = np.std(NWC_1)
  NWC_2_std = np.std(NWC_2)
  NSC_1_std = np.std(NSC_1)
  NSC_2_std = np.std(NSC_2)
  EEJ_1_std = np.std(EEJ_1)
  EEJ_2_std = np.std(EEJ_2)
  purity_1_std = np.std(purity_1)
  purity_2_std = np.std(purity_2)
  F_1_std = np.std(F_1)
  F_2_std = np.std(F_2)

  nfull = len(all_drawings)
  n1 = len(drawing_subset_1)
  n2 = len(drawing_subset_2)
    
  for category in categories:
    for network in [' 1',' 2']:
      category_strengths[category + network + ' mean'] = np.mean(category_strengths[category + network])
      category_strengths[category + network + ' std'] = np.std(category_strengths[category + network])
      category_internal_connections[category + network + ' mean'] = np.mean(category_internal_connections[category + network])
      category_internal_connections[category + network + ' std'] = np.std(category_internal_connections[category + network])
      category_NDCs[category + network + ' mean'] = np.mean(category_NDCs[category + network])
      category_NDCs[category + network + ' std'] = np.std(category_NDCs[category + network])
      category_NWCs[category + network + ' mean'] = np.mean(category_NWCs[category + network])
      category_NWCs[category + network + ' std'] = np.std(category_NWCs[category + network])
      category_NSCs[category + network + ' mean'] = np.mean(category_NSCs[category + network])
      category_NSCs[category + network + ' std'] = np.std(category_NSCs[category + network])
      category_betweennesses[category + network + ' mean'] = np.mean(category_betweennesses[category + network])
      category_betweennesses[category + network + ' std'] = np.std(category_betweennesses[category + network])
      category_closenesses[category + network + ' mean'] = np.mean(category_closenesses[category + network])
      category_closenesses[category + network + ' std'] = np.std(category_closenesses[category + network])
    category_strengths[category + ' d'] = CohensD(category_strengths[category + ' 1 mean'],category_strengths[category + ' 1 std'],n1,category_strengths[category + ' 2 mean'],category_strengths[category + ' 2 std'],n2)
    category_internal_connections[category + ' d'] = CohensD(category_internal_connections[category + ' 1 mean'],category_internal_connections[category + ' 1 std'],n1,category_internal_connections[category + ' 2 mean'],category_internal_connections[category + ' 2 std'],n2)
    category_NDCs[category + ' d'] = CohensD(category_NDCs[category + ' 1 mean'],category_NDCs[category + ' 1 std'],n1,category_NDCs[category + ' 2 mean'],category_NDCs[category + ' 2 std'],n2)
    category_NWCs[category + ' d'] = CohensD(category_NWCs[category + ' 1 mean'],category_NWCs[category + ' 1 std'],n1,category_NWCs[category + ' 2 mean'],category_NWCs[category + ' 2 std'],n2)
    category_NSCs[category + ' d'] = CohensD(category_NSCs[category + ' 1 mean'],category_NSCs[category + ' 1 std'],n1,category_NSCs[category + ' 2 mean'],category_NSCs[category + ' 2 std'],n2)
    category_betweennesses[category + ' d'] = CohensD(category_betweennesses[category + ' 1 mean'],category_betweennesses[category + ' 1 std'],n1,category_betweennesses[category + ' 2 mean'],category_betweennesses[category + ' 2 std'],n2)
    category_closenesses[category + ' d'] = CohensD(category_closenesses[category + ' 1 mean'],category_closenesses[category + ' 1 std'],n1,category_closenesses[category + ' 2 mean'],category_closenesses[category + ' 2 std'],n2)
    
  for node in big_nodes:
    for measure in [' betweenness',' closeness',' normnodedegree',' normnodestrength']:
      key = node + measure
      centralities_1[key + ' mean'] = np.average(centralities_1[key])
      centralities_1[key + ' std'] = np.std(centralities_1[key])
      centralities_2[key + ' mean'] = np.average(centralities_2[key])
      centralities_2[key + ' std'] = np.std(centralities_2[key])
    
  d_NDC = CohensD(NDC_1_mean,NDC_1_std,n1,NDC_2_mean,NDC_2_std,n2)
  d_NWC = CohensD(NWC_1_mean,NWC_1_std,n1,NWC_2_mean,NWC_2_std,n2)
  d_NSC = CohensD(NSC_1_mean,NSC_1_std,n1,NSC_2_mean,NSC_2_std,n2)
  d_EEJ = CohensD(EEJ_1_mean,EEJ_1_std,n1,EEJ_2_mean,EEJ_2_std,n2)
  d_purity = CohensD(purity_1_mean,purity_1_std,n1,purity_2_mean,purity_2_std,n2)
  d_F = CohensD(F_1_mean,F_1_std,n1,F_2_mean,F_2_std,n2)

  output = {'NDC1 mean':NDC_1_mean,'NDC2 mean':NDC_2_mean,
          'NWC1 mean':NWC_1_mean,'NWC2 mean':NWC_2_mean,
          'NSC1 mean':NSC_1_mean,'NSC2 mean':NSC_2_mean,
          'EEJ1 mean':EEJ_1_mean,'EEJ2 mean':EEJ_2_mean,
          'purity1 mean':purity_1_mean,'purity2 mean':purity_2_mean,
          'F1 mean':F_1_mean,'F2 mean':F_2_mean,
          'NDC1 std':NDC_1_std,'NDC2 std':NDC_2_std,
          'NWC1 std':NWC_1_std,'NWC2 std':NWC_2_std,
          'NSC1 std':NSC_1_std,'NSC2 std':NSC_2_std,
          'EEJ1 std':EEJ_1_std,'EEJ2 std':EEJ_2_std,
          'purity1 std':purity_1_std,'purity2 std':purity_2_std,
          'F1 std':F_1_std,'F2 std':F_2_std,
          'd NDC':d_NDC,'d NWC':d_NWC,'d NSC':d_NSC,'d EEJ':d_EEJ,'d purity':d_purity, 'd FMeasure':d_F}
    
  for node in big_nodes:
#     for measure in ['betweenness','closeness','eigencentrality']:
    for measure in [' betweenness',' closeness',' normnodedegree',' normnodestrength']:
      key = node + measure 
      output[key + ' d'] = CohensD(centralities_1[key + ' mean'],centralities_1[key + ' std'],n1,centralities_2[key + ' mean'],centralities_2[key + ' std'],n2)
      output[key + ' p'] = f_oneway(centralities_1[key], centralities_2[key]).pvalue
      output[key + ' 1 mean'] = centralities_1[key + ' mean']
      output[key + ' 2 mean'] = centralities_2[key + ' mean']
      output[key + ' 1 std'] = centralities_1[key + ' std']
      output[key + ' 2 std'] = centralities_2[key + ' std']
        
  print('--Bootstrap Comparison Completed--')
  print('Highest-frequency nodes:')
  for node in big_nodes:
    print(node,'     Full count:',      G_full.nodes[node]['weight'],' (',      G_full.nodes[node]['weight']/nfull*100,'%) ')
    print(node,' ',subset_name_1,' count:',G_1_original.nodes[node]['weight'],' (',G_1_original.nodes[node]['weight']/n1   *100,'%) ')
    print(node,' ',subset_name_2,' count:',G_2_original.nodes[node]['weight'],' (',G_2_original.nodes[node]['weight']/n2   *100,'%) ')
  print('First Subset')
  print('     NDC ',NDC_1_mean,'±',NDC_1_std)
  print('     NWC ',NWC_1_mean,'±',NWC_1_std)
  print('     NSC ',NSC_1_mean,'±',NSC_1_std)
  print('     EEJ ',EEJ_1_mean,'±',EEJ_1_std)
  print('  purity ',purity_1_mean,'±',purity_1_std)
  print('FMeasure ',F_1_mean,'±',F_1_std)
  print('Second Subset')
  print('     NDC ',NDC_2_mean,'±',NDC_2_std)
  print('     NWC ',NWC_2_mean,'±',NWC_2_std)
  print('     NSC ',NSC_2_mean,'±',NSC_2_std)
  print('     EEJ ',EEJ_2_mean,'±',EEJ_2_std)
  print('  purity ',purity_2_mean,'±',purity_2_std)
  print('FMeasure ',F_2_mean,'±',F_2_std)
  print('Cohen''s d effect sizes')
  print('     NDC',d_NDC)
  print('     NWC',d_NWC)
  print('     NSC',d_NSC)
  print('     EEJ',d_EEJ)
  print('  purity',d_purity)
  print('FMeasure',d_F)
  for key, value in output.items():
    print(key + ' : ' + str(value))

  with open(file_out, 'w') as convert_file: 
    convert_file.write('FINISHED at ' + str(datetime.datetime.now()))
    convert_file.write('\n')
    convert_file.write('N = '+str(N))
    convert_file.write('\n')
    convert_file.write('Subsets '+subset_name_1+'; '+subset_name_2)
    convert_file.write('\n')
    convert_file.write('N_nodes = '+str(N_nodes))
    convert_file.write('\n')
    convert_file.write('centrality_power = '+str(centrality_power))
    convert_file.write('\n')
    convert_file.write('clustering_method = '+clustering_method)
    convert_file.write('\n')
    convert_file.write('min_node_weight = '+str(min_node_weight))
    convert_file.write('\n')
    convert_file.write('--Bootstrap Comparison Completed--\n\n')

    # Write LaTeX-formatted table of centrality measures.
    header1 = 'Drawing & \multicolumn{2}{c|}{Frequency ($\%$)}'
    header2 = 'Element (Category) & ' + subset_name_1 + ' & ' + subset_name_2
    if 'betweenness' in measures_in_table:
        header1 += '& \multicolumn{3}{c|}{Betweenness} '
        header2 += ' & ' + subset_name_1 + ' & ' + subset_name_2 + ' & $d$ '
    if 'normdegree' in measures_in_table:
        header1 += '& \multicolumn{3}{c|}{Normalized Degree} '
        header2 += ' & ' + subset_name_1 + ' & ' + subset_name_2 + ' & $d$ '
    if 'normstrength' in measures_in_table:
        header1 += '& \multicolumn{3}{c|}{Normalized Strength} '
        header2 += ' & ' + subset_name_1 + ' & ' + subset_name_2 + ' & $d$ '
    header1 += '\\\\\n'
    header2 += '\\\\\n'
    convert_file.write(header1)
    convert_file.write(header2)
#     convert_file.write('Drawing & \multicolumn{2}{c|}{Frequency ($\%$)} & \multicolumn{3}{c|}{Betweenness} & \multicolumn{3}{c|}{Normalized Degree} & \multicolumn{3}{c|}{Normalized Strength} \\\\\n')
#     header = 
#     convert_file.write('Element (Category) & ' + subset_name_1 + ' & ' + subset_name_2 + ' & '+ subset_name_1 + ' & ' + subset_name_2 + ' & $d$ & ' + subset_name_1 + ' & ' + subset_name_2 + ' & $d$ & ' + subset_name_1 + ' & ' + subset_name_2 + ' & $d$ \\\\\n' )
    convert_file.write('\\hline\n')
    for node in big_nodes:
      to_write = node+' ('+G_1_original.nodes.data()[node]['category']+') & $' + NumberOut(G_1_original.nodes[node]['weight']/n1*100,1)+' $ & $' + NumberOut(G_2_original.nodes[node]['weight']/n2*100,1) + '$'
      if 'betweenness' in measures_in_table:
        to_write += ' & $' + Uncertainty(output[node + ' betweenness 1 mean'],output[node + ' betweenness 1 std'],num_sig) + '$ & $' + Uncertainty(output[node + ' betweenness 2 mean'],output[node + ' betweenness 2 std'],num_sig) + ' $ & $' + NumberOut(output[node + ' betweenness d'],num_sig-1)     +CohensStar(output[node + ' betweenness d'     ])+'$'
      if 'normdegree' in measures_in_table:
        to_write += ' & $' + Uncertainty(output[node + ' normdegree 1 mean'],output[node + ' normdegree 1 std'],num_sig) + '$ & $' + Uncertainty(output[node + ' normdegree 2 mean'],output[node + ' normdegree 2 std'],num_sig) + ' $ & $' + NumberOut(output[node + ' normdegree d'],num_sig-1)     +CohensStar(output[node + ' normdegree d'     ])+'$'
      if 'normstrength' in measures_in_table:
        to_write += ' & $' + Uncertainty(output[node + ' normstrength 1 mean'],output[node + ' normstrength 1 std'],num_sig) + '$ & $' + Uncertainty(output[node + ' normstrength 2 mean'],output[node + ' normstrength 2 std'],num_sig) + ' $ & $' + NumberOut(output[node + ' normstrength d'],num_sig-1)     +CohensStar(output[node + ' normstrength d'     ])+'$'
      to_write += '\\\\\n'
      convert_file.write(to_write)
        
        
#       convert_file.write(WriteBootStrapComparisonTableLine(node+' ('+G_1_original.nodes.data()[node]['category']+') ', NumberOut(G_1_original.nodes[node]['weight']/n1*100,1),NumberOut(G_2_original.nodes[node]['weight']/n2*100,1),
#                                                            '$'+Uncertainty(output[node + ' betweenness 1 mean'],     output[node + ' betweenness 1 std'],num_sig)     +'$','$'+Uncertainty(output[node + ' betweenness 2 mean'],     output[node + ' betweenness 2 std'],num_sig)     +'$','$'+NumberOut(output[node + ' betweenness d'],num_sig-1)     +CohensStar(output[node + ' betweenness d'     ])+'$',
#                                                            '$'+Uncertainty(output[node + ' normnodedegree 1 mean'],  output[node + ' normnodedegree 1 std'],num_sig)  +'$','$'+Uncertainty(output[node + ' normnodedegree 2 mean'],  output[node + ' normnodedegree 2 std'],num_sig)  +'$','$'+NumberOut(output[node + ' normnodedegree d'],num_sig-1)  +CohensStar(output[node + ' normnodedegree d'  ])+'$',
#                                                            '$'+Uncertainty(output[node + ' normnodestrength 1 mean'],output[node + ' normnodestrength 1 std'],num_sig)+'$','$'+Uncertainty(output[node + ' normnodestrength 2 mean'],output[node + ' normnodestrength 2 std'],num_sig)+'$','$'+NumberOut(output[node + ' normnodestrength d'],num_sig-1)+CohensStar(output[node + ' normnodestrength d'])+'$'))

    convert_file.write('\n')
    convert_file.write('Highest-frequency nodes:\n')
    for node in big_nodes:
      convert_file.write(node+'     Full count:'+      str(G_full.nodes[node]['weight'])+' ('+      str(G_full.nodes[node]['weight']/nfull*100)+'%) \n')
      convert_file.write(node+' '+subset_name_1+' count:'+str(G_1_original.nodes[node]['weight'])+' ('+str(G_1_original.nodes[node]['weight']/n1   *100)+'%) \n')
      convert_file.write(node+' '+subset_name_2+' count:'+str(G_2_original.nodes[node]['weight'])+' ('+str(G_2_original.nodes[node]['weight']/n2   *100)+'%) \n')
    convert_file.write('####################################')
    convert_file.write('\n')
    convert_file.write(subset_name_1+' (N='+str(n1)+')')
    convert_file.write('\n')
    convert_file.write(str(G_1_original.elements_per_drawing) + ' nodes per drawing')
    convert_file.write('\n')
    convert_file.write('     NDC '+str(NDC_1_mean)+' ± '+str(NDC_1_std))
    convert_file.write('\n')
    convert_file.write('     NWC '+str(NWC_1_mean)+' ± '+str(NWC_1_std))
    convert_file.write('\n')
    convert_file.write('     NSC '+str(NSC_1_mean)+' ± '+str(NSC_1_std))
    convert_file.write('\n')
    convert_file.write('     EEJ '+str(EEJ_1_mean)+' ± '+str(EEJ_1_std))
    convert_file.write('\n')
    convert_file.write('  purity '+str(purity_1_mean)+' ± '+str(purity_1_std))
    convert_file.write('\n')
    convert_file.write('FMeasure '+str(F_1_mean)+' ± '+str(F_1_std))
    convert_file.write('\n')
    convert_file.write('####################################')
    convert_file.write('\n')
    convert_file.write(subset_name_2+' (N='+str(n2)+')')
    convert_file.write('\n')
    convert_file.write(str(G_2_original.elements_per_drawing) + ' nodes per drawing')
    convert_file.write('\n')
    convert_file.write('     NDC '+str(NDC_2_mean)+' ± '+str(NDC_2_std))
    convert_file.write('\n')
    convert_file.write('     NWC '+str(NWC_2_mean)+' ± '+str(NWC_2_std))
    convert_file.write('\n')
    convert_file.write('     NSC '+str(NSC_2_mean)+' ± '+str(NSC_2_std))
    convert_file.write('\n')
    convert_file.write('     EEJ '+str(EEJ_2_mean)+' ± '+str(EEJ_2_std))
    convert_file.write('\n')
    convert_file.write('  purity '+str(purity_2_mean)+' ± '+str(purity_2_std))
    convert_file.write('\n')
    convert_file.write('FMeasure '+str(F_2_mean)+' ± '+str(F_2_std))
    convert_file.write('\n')
    convert_file.write('####################################')
    convert_file.write('\n')
    convert_file.write('Cohen''s d effect sizes')
    convert_file.write('\n')
    convert_file.write('     NDC '+str(d_NDC))
    convert_file.write('\n')
    convert_file.write('     NWC '+str(d_NWC))
    convert_file.write('\n')
    convert_file.write('     NSC '+str(d_NSC))
    convert_file.write('\n')
    convert_file.write('     EEJ '+str(d_EEJ))
    convert_file.write('\n')
    convert_file.write('  purity '+str(d_purity))
    convert_file.write('\n')
    convert_file.write('FMeasure '+str(d_F))
    convert_file.write('\n')
    convert_file.write('####################################')
    convert_file.write('\n')

    convert_file.write('=== Within-Category Measures ===' + '\n')
    print('=== Within-Category Measures ===')
    for category in categories:
      convert_file.write(category + '\n')
      print(category)
      for network in [' 1',' 2']:
        convert_file.write('Subset ' + network + '\n')
        print('Subset',network)
        convert_file.write('weights' + ' '+str(category_weights[category + network]) + '\n')
        print('weights',category_weights[category + network])
        for dictionary in [category_strengths,category_internal_connections,category_NDCs,
                           category_NWCs,category_NSCs,category_betweennesses,category_closenesses]:
          convert_file.write(dictionary['measure'] + ' '+str(dictionary[category + network + ' mean']) + ' ± ' + str(dictionary[category + network + ' std']) + ', d = ' + str(dictionary[category + ' d']) + '\n')
          print(dictionary['measure'], ' ',dictionary[category + network + ' mean'], ' ± ' , dictionary[category + network + ' std'] , ', d = ' , dictionary[category + ' d'])
    convert_file.write('####################################')
    convert_file.write('\n')
    convert_file.write('            All Output')
    convert_file.write('\n')
    convert_file.write('####################################')
    convert_file.write('\n')
    
    for key, value in output.items():
      convert_file.write(key + ' : ' + str(value))
      convert_file.write('\n')
    

  return output

def NumberOut(x,num_sig=3):
  # Return a string representation of float x with n_sig decimal places, including zeros.
  # return '{:.'+str(n_sig)+'f}'.format(np.round(x,n_sig))
  out = '{x:.'+str(num_sig)+'f}'
  return out.format(x=x)

def WriteBootStrapComparisonTableLine(row_header,*strings):
#   print('in WriteBootStrapComparisonTableLine for ',row_header)
  table_line = row_header
  for i in range(len(strings)):
#     print(strings[i])
    table_line += ' & ' + strings[i]
  table_line += '\\\\\n'
  return table_line

def Uncertainty(x,dx,num_sig):
    str_x = NumberOut(x,num_sig)
    str_dx = NumberOut(dx,num_sig)
    i_start = len(str_dx)
    i = 0
    while i < len(str_dx) and i_start == len(str_dx):
        if (str_dx[i] != '0') and (str_dx[i] != '.'):
            i_start = i
        i += 1
    return str_x + '(' + str_dx[i_start:] + ')'

def CompareBootstrapStructures(G,N,nodes_in,outfile='dendro_status.txt'):
    # Create N bootstraps of graph G.
    # Create a dendrogram of nodes_in from each bootstrap.
    # Return the number of unique community structures at each dendrogram level l.
    
    # Create list of community structures.
    bootstrap_comms = []
    
    # Create bootstraps.
    for n in range(0,N):
        print(n)
        with open(outfile, 'w') as convert_file: 
            current_time = datetime.datetime.now()
            convert_file.write(str(current_time) + ' bootstrap ' + str(n+1) + ' of ' + str(N))
        G_bootstrap = MakeBootstrapGraph(G)
        # Extract communities.
        bootstrap_comms.append(FilterCommunities(list(nx.community.girvan_newman(G_bootstrap,most_valuable_edge=most_central_edge)),nodes_in))
        structures_count,structures_histogram = CountCommunities(bootstrap_comms)
    return structures_count,structures_histogram,bootstrap_comms

def CountCommunities(comms):
    # comms[n] is the nth list of list of lists, with comms[n][l] the lth community structure in comms[n]
    # Count how many unique community structures there are at each level l of comms[n]'s dendrogram
    # Returns structures_count = a list of the counts of unique community structures at each level l
    # and structures_histogram = a list of lists of the number of times each structure was found at level l.
    # How many levels deep do we need to go?
    lmin = 1000000000
    for n in range(len(comms)):
        lmin = min(lmin,len(comms[n]))
#     print(lmin)
    
    structures_count     = []
    structures_histogram = []
    
    # Loop over dendrogram level.
#     convert_file = open('dendro_status.txt', 'w')
    for l in range(0,lmin):
        current_time = datetime.datetime.now()
#         convert_file.write(str(current_time) + ' dendrogram level ' + str(l+1) + ' of ' + str(lmin))
#         convert_file.write('\n')
        # Create a list to track the community structures that have been found at this level.
        comms_found = []
        histogram   = []
        # Loop over comms.
        for n in range(0,len(comms)):
#             convert_file.write('In comm number'+str(n))
#             convert_file.write('\n')

            # Check for whether s = comms[n][l] is already in comms_found, ignoring permutations.
            found_s = False
            for comm in comms_found:
                s = comms[n][l].copy()
                to_remove = []
                for element in s:
                    if element == [] or element in comm:
                        to_remove.append(element)
                for element in to_remove:
                    s.remove(element)
#                 convert_file.write('I ended with len(s)=',len(s))
                if len(s) == 0:
                    found_s = True
                    comm_index = comms_found.index(comm)
            if found_s:
                histogram[comm_index] += 1
            else:
                comms_found.append(comms[n][l])
                histogram.append(1)
#         print(l,len(comms_found))
        structures_count.append(len(comms_found))
        structures_histogram.append(histogram)    
    return structures_count,structures_histogram

def FilterCommunities(comms_in,nodes_in):
  # comms_in is the list of list of sets from girvan_newman
  # comms_out is this list of list of lists with only nodes_in
  comms_out = []
  for comm_list_in in comms_in:
    comm_list_out = []
    for comm_in in comm_list_in:
      comm_out = []
      for node in comm_in:
        if node in nodes_in: comm_out.append(node)
      comm_list_out.append(comm_out)
    comms_out.append(comm_list_out)
  return comms_out

def GraphCommunityStructures(bootstrap_structures_histogram):
  width = 0.5
  fig, ax = plt.subplots(figsize=(20,10))
  for l in range(len(bootstrap_structures_histogram)):
    bottom = 0
    for h in bootstrap_structures_histogram[l]:
      p = ax.bar(l, h, width, label=False, bottom=bottom)
      bottom += h
  ax.set_title("Number of Bootstrapped Community Structures at Each Dendrogram Level")
  plt.show()
  return

def GetBigNodes(G,N_nodes):
    # Return a list of the N_nodes highest-weight nodes from G.
    return sorted(G.nodes, key=lambda x: G.nodes[x]['weight'], reverse=True)[0:N_nodes]

def PruneLargeNodes(G,N_nodes):
    # Return a copy of G with the N_nodes highest-weight nodes removed, and the list of nodes that were removed.
    G_pruned = deepcopy(G)
    big_nodes = GetBigNodes(G_pruned,N_nodes)
    G_pruned.remove_nodes_from(big_nodes)
    return G_pruned,big_nodes


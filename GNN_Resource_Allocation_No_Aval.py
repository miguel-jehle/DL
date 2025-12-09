# ==============================================================================
# MODIFIED CODE - SKIPS GNN_AVAL TRAINING
# ==============================================================================

# %% [Same cells 1-6, but skip training in cell 6]

# %%
# ==============================================================================
# CÉLULA 1: Instalação e Configuração de Imports
# ==============================================================================
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports do PyTorch Geometric (PyG)
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, global_mean_pool

# Outras bibliotecas
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import math
from tqdm import tqdm

print(f"PyTorch Version: {torch.__version__}")
print(f"PyG Version: {torch_geometric.__version__}")

# Configurações
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %%
# ==============================================================================
# CÉLULA 2: Geração de Dados Sintéticos (Problema)
# ==============================================================================
#
# Precisamos de duas coisas para um "problema":
# 1. Um workflow (DAG) de tarefas.
# 2. Uma matriz de custo/tempo (Tarefas x VMs).
#

def generate_workflow_dag(num_tasks: int, max_deps: int = 2, density: float = 0.15):
    """
    Gera um DAG de workflow aleatório usando NetworkX.
    Garante que é um DAG ao adicionar arestas apenas "para frente".
    """
    G = nx.DiGraph()
    nodes = list(range(num_tasks))
    random.shuffle(nodes) # Garante uma ordem topológica aleatória

    for i in range(1, num_tasks):
        node_i = nodes[i]
        possible_predecessors = nodes[:i]

        # Garante que cada tarefa (exceto a primeira) tenha pelo menos um predecessor
        # para criar um grafo mais conectado, mas isso é opcional.
        # Aqui, vamos focar na densidade.

        num_deps = 0
        for j in range(len(possible_predecessors)):
            if num_deps >= max_deps:
                break
            if random.random() < density:
                node_j = possible_predecessors[j]
                G.add_edge(node_j, node_i) # Aresta vai do predecessor para o sucessor
                num_deps += 1

    # Garante que o grafo é conectado (opcional, mas bom)
    # Vamos simplificar e assumir que a densidade cuida disso por enquanto.

    # Adiciona um nó "source" e "sink" para garantir início e fim únicos
    source = "source"
    sink = "sink"
    G.add_node(source)
    G.add_node(sink)

    roots = [node for node, in_degree in G.in_degree() if in_degree == 0 and node not in [source, sink]]
    leaves = [node for node, out_degree in G.out_degree() if out_degree == 0 and node not in [source, sink]]

    for root in roots:
        G.add_edge(source, root)

    for leaf in leaves:
        G.add_edge(leaf, sink)

    # Renomeia nós para inteiros para facilitar o PyG
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    return G

def generate_cost_time_matrix(num_tasks: int, num_vms: int):
    """
    Gera matrizes aleatórias de tempo e custo.
    """
    # Tempo: e.g., de 10 a 100 unidades
    times = np.random.randint(10, 101, size=(num_tasks, num_vms)).astype(np.float32)
    # Custo: e.g., de 1 a 20 unidades
    costs = np.random.randint(1, 21, size=(num_tasks, num_vms)).astype(np.float32)
    return times, costs

# --- Teste das funções ---
NUM_TASKS_TEST = 8
NUM_VMS_TEST = 2

test_dag = generate_workflow_dag(NUM_TASKS_TEST)
# Note: O DAG terá NUM_TASKS_TEST + 2 nós (source, sink)
num_nodes_total = test_dag.number_of_nodes()
print(f"Número total de nós no DAG (incluindo source/sink): {num_nodes_total}")

test_times, test_costs = generate_cost_time_matrix(num_nodes_total, NUM_VMS_TEST)
print(f"Shape da Matriz de Tempos: {test_times.shape}")
print(f"Shape da Matriz de Custos: {test_costs.shape}")

# Visualização do DAG
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(test_dag)
nx.draw(test_dag, pos, with_labels=True, node_color='lightblue', font_weight='bold')
plt.title("Exemplo de DAG de Workflow")
plt.show()

# %%
# ==============================================================================
# CÉLULA 3: O Simulador (Ground Truth)
# ==============================================================================
#
# Esta é uma função crucial. Ela NÃO é uma GNN.
# É um simulador que calcula o Makespan (C_max) e Custo Total (C_total)
# para uma DADA alocação.
# Usaremos isso para treinar a GNN_Aval.
#

def calculate_metrics(dag: nx.DiGraph,
                      allocations: dict,
                      time_matrix: np.ndarray,
                      cost_matrix: np.ndarray):
    """
    Calcula o makespan (C_max) e o custo total (C_total) para uma alocação.

    Args:
        dag: O grafo do workflow (NetworkX).
        allocations: Dicionário {node_id: vm_id}.
        time_matrix: Matriz (num_tasks, num_vms) de tempos.
        cost_matrix: Matriz (num_tasks, num_vms) de custos.
    """

    # Garante que o grafo é um DAG
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("O grafo não é um DAG!")

    try:
        topo_sort = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        # Se o grafo tiver ciclos (não deveria), isso falhará.
        print("Erro: Grafo contém ciclos.")
        return float('inf'), float('inf')

    finish_times = {}
    total_cost = 0.0

    for node in topo_sort:
        # Obter tempos de término dos predecessores
        pred_finish_times = [finish_times.get(pred, 0) for pred in dag.predecessors(node)]

        start_time = max(pred_finish_times) if pred_finish_times else 0

        # Obter VM alocada
        vm_id = allocations.get(node) # e.g., {0: 1, 1: 0, 2: 1, ...}

        if vm_id is None:
            # Nó não foi alocado? Isso é um erro.
            # Para nós 'source' e 'sink', o tempo/custo é 0.
            if node in [topo_sort[0], topo_sort[-1]]: # Assumindo source=0, sink=último
                 exec_time = 0.0
                 exec_cost = 0.0
            else:
                 raise ValueError(f"Nó {node} não encontrado nas alocações.")
        else:
            exec_time = time_matrix[node, vm_id]
            exec_cost = cost_matrix[node, vm_id]

        finish_times[node] = start_time + exec_time
        total_cost += exec_cost

    makespan = finish_times[topo_sort[-1]] # Makespan é o tempo de término do nó 'sink'

    return makespan, total_cost

# --- Heurísticas para o pré-treino da GNN_Aval ---
def heuristic_fastest(dag, time_m):
    alloc_dict = {}
    for node_id in dag.nodes():
        vm_id = np.argmin(time_m[node_id])
        alloc_dict[node_id] = vm_id
    return alloc_dict

def heuristic_cheapest(dag, cost_m):
    alloc_dict = {}
    for node_id in dag.nodes():
        vm_id = np.argmin(cost_m[node_id])
        alloc_dict[node_id] = vm_id
    return alloc_dict

# --- Teste do Simulador ---
# Cria uma alocação aleatória
num_nodes_total = test_dag.number_of_nodes()
random_alloc = {node_id: random.randint(0, NUM_VMS_TEST - 1) for node_id in test_dag.nodes()}

# Trata source/sink
source_node = list(nx.topological_sort(test_dag))[0]
sink_node = list(nx.topological_sort(test_dag))[-1]
random_alloc[source_node] = 0 # Não importa, custo/tempo será 0
random_alloc[sink_node] = 0

# Corrige matrizes para terem custo/tempo 0 para source/sink
test_times[source_node, :] = 0
test_times[sink_node, :] = 0
test_costs[source_node, :] = 0
test_costs[sink_node, :] = 0


makespan, total_cost = calculate_metrics(test_dag, random_alloc, test_times, test_costs)
print(f"Simulador (Alocação Aleatória):")
print(f"  - Makespan (C_max): {makespan:.2f}")
print(f"  - Custo Total (C_total): {total_cost:.2f}")

# %%
# ==============================================================================
# CÉLULA 4: Conversão para Grafo Heterogêneo (PyG)
# ==============================================================================
#
# Esta é a representação de dados que as GNNs irão consumir.
# Vamos usar a estrutura HeteroData do PyG.
#
# Tipos de Nós:
# - 'task': Tarefas do workflow (incluindo source/sink)
# - 'vm': Máquinas Virtuais
#
# Tipos de Arestas:
# - ('task', 'depends_on', 'task'): Arestas do DAG
# - ('task', 'can_run_on', 'vm'): Arestas de alocação (grafo bipartido)
#

def create_pyg_data(dag: nx.DiGraph,
                      time_matrix: np.ndarray,
                      cost_matrix: np.ndarray):
    """
    Cria um objeto HeteroData do PyG para um problema.
    """
    data = HeteroData()

    num_tasks = dag.number_of_nodes()
    num_vms = time_matrix.shape[1]

    time_m = time_matrix.copy()
    cost_m = cost_matrix.copy()

    # Evitar divisão por zero se houver apenas 1 VM
    if num_vms > 1:
        time_mean = time_m.mean(axis=1, keepdims=True)
        time_std = time_m.std(axis=1, keepdims=True)
        cost_mean = cost_m.mean(axis=1, keepdims=True)
        cost_std = cost_m.std(axis=1, keepdims=True)
    else:
        time_mean = time_m
        time_std = np.zeros_like(time_m)
        cost_mean = cost_m
        cost_std = np.zeros_like(cost_m)

    time_min = time_m.min(axis=1, keepdims=True)
    cost_min = cost_m.min(axis=1, keepdims=True)

    # 6 features por tarefa
    task_features = np.concatenate([
        time_mean, time_std, time_min,
        cost_mean, cost_std, cost_min
    ], axis=1).astype(np.float32)

    # Normalizar as features
    task_features = (task_features - task_features.mean(axis=0)) / (task_features.std(axis=0) + 1e-6)

    data['task'].x = torch.from_numpy(task_features)

    # Características iniciais dos nós 'vm' (ainda podem ser zeros)
    data['vm'].x = torch.zeros((num_vms, 4)) # Embedding inicial de 4 dimensões

    # --- Arestas de Dependência ('task' -> 'task') ---
    edge_list = list(dag.edges()) # Lista de tuplas (u, v)
    if edge_list:
        edge_index_deps = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index_deps = torch.empty((2, 0), dtype=torch.long)
    data['task', 'depends_on', 'task'].edge_index = edge_index_deps

    # --- Arestas de Alocação ('task' -> 'vm') ---
    # Grafo bipartido completo: toda tarefa pode rodar em toda VM
    task_indices = torch.arange(num_tasks).repeat_interleave(num_vms)
    vm_indices = torch.arange(num_vms).repeat(num_tasks)
    edge_index_alloc = torch.stack([task_indices, vm_indices], dim=0)

    data['task', 'can_run_on', 'vm'].edge_index = edge_index_alloc

    # --- Atributos das Arestas de Alocação ---
    # As características mais importantes! (Tempo e Custo)
    # Precisamos achatar as matrizes na ordem correta
    edge_attr_time = torch.from_numpy(time_matrix).float().reshape(-1, 1)
    edge_attr_cost = torch.from_numpy(cost_matrix).float().reshape(-1, 1)

    # Normalização dos atributos (crucial para GNNs)
    edge_attr_time = (edge_attr_time - edge_attr_time.mean()) / (edge_attr_time.std() + 1e-6)
    edge_attr_cost = (edge_attr_cost - edge_attr_cost.mean()) / (edge_attr_cost.std() + 1e-6)

    data['task', 'can_run_on', 'vm'].edge_attr = torch.cat([edge_attr_time, edge_attr_cost], dim=1)

    # --- Arestas Reversas (para agregação de mensagens) ---
    # PyG HeteroConv precisa que arestas reversas sejam definidas
    data['vm', 'rev_can_run_on', 'task'].edge_index = edge_index_alloc.flip(0)
    data['vm', 'rev_can_run_on', 'task'].edge_attr = data['task', 'can_run_on', 'vm'].edge_attr

    data['task', 'rev_depends_on', 'task'].edge_index = edge_index_deps.flip(0)

    return data

# --- Teste da Conversão ---
pyg_data = create_pyg_data(test_dag, test_times, test_costs)
print("\nObjeto PyG HeteroData:")
print(pyg_data)

# Validar a estrutura
print(f"\nTipos de Nós: {pyg_data.node_types}")
print(f"Tipos de Arestas: {pyg_data.edge_types}")
print(f"Arestas 'depends_on': {pyg_data['task', 'depends_on', 'task'].edge_index.shape}")
print(f"Arestas 'can_run_on': {pyg_data['task', 'can_run_on', 'vm'].edge_index.shape}")
print(f"Atributos 'can_run_on': {pyg_data['task', 'can_run_on', 'vm'].edge_attr.shape}")

# %%
# ==============================================================================
# CÉLULA 5: Modelo 1 - GNN de Avaliação (GNN_Aval)
# ==============================================================================
#
# Esta GNN aprende a PREVER o Makespan e o Custo Total.
# Ela atua como um "simulador diferenciável".
#
# Entrada: Um grafo PyG do problema.
#          + Características da *alocação escolhida* adicionadas aos nós 'task'.
# Saída: Um escalar para 'makespan' e um para 'cost_total'.
#

class GNN_Aval(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers=4, heads = 4):
        super().__init__()

        # Camada de entrada para 'task'.
        # MELHORIA: A entrada agora é (6 features + 2 da alocação) = 8
        self.task_emb = nn.Linear(8, hidden_dim) # 6 (features) + 2 (tempo, custo)
        self.vm_emb = nn.Linear(4, hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers): # Número de camadas configurável
            conv = HeteroConv({
                ('task', 'depends_on', 'task'): GATConv(-1, hidden_dim // heads, heads = heads),
                ('task', 'rev_depends_on', 'task'): GATConv(-1 , hidden_dim // heads, heads = heads),
                ('task', 'can_run_on', 'vm'): GATConv((-1, -1) , hidden_dim // heads, heads = heads, add_self_loops=False),
                ('vm', 'rev_can_run_on', 'task'): GATConv((-1, -1), hidden_dim // heads, heads = heads, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        # MLP de Saída (prediz 2 valores)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim) # out_dim = 2 (makespan, cost)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1. Aplicar embeddings de entrada
        x_dict['task'] = self.task_emb(x_dict['task']).relu()
        x_dict['vm'] = self.vm_emb(x_dict['vm']).relu()

        # 2. Camadas de Convolução
        for conv in self.convs:
            x_dict_update = conv(x_dict, edge_index_dict, edge_attr_dict)
            # Atualiza características (com conexões residuais)
            x_dict = {key: x_dict[key] + x_dict_update[key].relu() for key in x_dict.keys()}

        # 3. Pooling Global
        # Usamos o pool apenas dos nós 'task' para a predição global
        task_pool = global_mean_pool(x_dict['task'], batch=None) # Assume batch_size=1

        # 4. MLP de Saída
        return self.out_mlp(task_pool)

# --- Teste do Modelo (Instanciação) ---
# Dimensões de embedding
HIDDEN_DIM = 64  # Reduzido de 256 para 64
NUM_LAYERS = 4
HEADS = 4
# Dimensão de saída (makespan, cost_total)
OUT_DIM = 2

gnn_aval_model = GNN_Aval(hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYERS, heads=HEADS)
print("\nEstrutura GNN_Aval:")
print(gnn_aval_model)

# Teste de forward (com dados dummy)
# Precisamos simular a entrada (x_dict com tempo/custo adicionados)
dummy_data = create_pyg_data(test_dag, test_times, test_costs)
num_tasks = dummy_data['task'].x.shape[0]

# Simula adição de tempo/custo da alocação
dummy_alloc_time = torch.randn(num_tasks, 1)
dummy_alloc_cost = torch.randn(num_tasks, 1)
# A entrada agora tem 6 features base + 2 de alocação
dummy_data['task'].x = torch.cat([dummy_data['task'].x, dummy_alloc_time, dummy_alloc_cost], dim=1)

# Passa pelo modelo
pred = gnn_aval_model(dummy_data.x_dict, dummy_data.edge_index_dict, dummy_data.edge_attr_dict)
print(f"\nSaída Dummy GNN_Aval (shape): {pred.shape}")

# %%
# ==============================================================================
# CÉLULA 6: SKIP TRAINING - JUST LOAD PRETRAINED MODELS
# ==============================================================================
#

class TargetNormalizer:
    """Classe simples para normalizar os alvos (makespan, custo)"""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.var = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.count = 0

    def update(self, y_true_batch):
        # y_true_batch é (N, 2)
        batch_mean = torch.mean(y_true_batch, dim=0)
        batch_var = torch.var(y_true_batch, dim=0)

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var

        self.count += y_true_batch.shape[0]

    def normalize(self, y):
        return (y - self.mean) / (torch.sqrt(self.var) + 1e-6)

    def denormalize(self, y_norm):
        return y_norm * (torch.sqrt(self.var) + 1e-6) + self.mean

    # ADICIONA state_dict
    def state_dict(self):
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }

    # ADICIONA load_state_dict
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']

def heuristic_random(dag, num_vms):
    alloc_dict = {}
    for node_id in dag.nodes():
        vm_id = random.randint(0, num_vms - 1)
        alloc_dict[node_id] = vm_id
    return alloc_dict

print("\nLoading pre-trained GNN_Aval and normalizer...")

# Initialize GNN_Aval model
HIDDEN_DIM = 64
NUM_LAYERS = 4
HEADS = 4
OUT_DIM = 2

gnn_aval = GNN_Aval(hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYERS, heads=HEADS)

# Load pre-trained weights
gnn_aval.load_state_dict(torch.load("gnn_aval_pretrained.pth"))
print("GNN_Aval model loaded from 'gnn_aval_pretrained.pth'")

# Load normalizer
target_normalizer = TargetNormalizer()
target_normalizer.load_state_dict(torch.load("gnn_aval_normalizer.pth"))
print("Normalizer loaded from 'gnn_aval_normalizer.pth'")

# Test the loaded model quickly
print("\nQuick test of loaded GNN_Aval...")
with torch.no_grad():
    # Create a dummy sample
    num_tasks = 10
    num_vms = 3
    dag = generate_workflow_dag(num_tasks)
    num_nodes = dag.number_of_nodes()
    time_m, cost_m = generate_cost_time_matrix(num_nodes, num_vms)
    
    source_node = list(nx.topological_sort(dag))[0]
    sink_node = list(nx.topological_sort(dag))[-1]
    time_m[source_node, :] = 0
    time_m[sink_node, :] = 0
    cost_m[source_node, :] = 0
    cost_m[sink_node, :] = 0
    
    # Create random allocation
    alloc_dict = heuristic_random(dag, num_vms)
    alloc_dict[source_node] = 0
    alloc_dict[sink_node] = 0
    
    # Calculate real metrics
    real_makespan, real_cost = calculate_metrics(dag, alloc_dict, time_m, cost_m)
    
    # Create PyG data with allocation
    pyg_data = create_pyg_data(dag, time_m, cost_m)
    
    alloc_times = torch.zeros(num_nodes, 1)
    alloc_costs = torch.zeros(num_nodes, 1)
    for node_id, vm_id in alloc_dict.items():
        alloc_times[node_id] = torch.tensor(time_m[node_id, vm_id])
        alloc_costs[node_id] = torch.tensor(cost_m[node_id, vm_id])
    
    # Normalize
    alloc_times = (alloc_times - alloc_times.mean()) / (alloc_times.std() + 1e-6)
    alloc_costs = (alloc_costs - alloc_costs.mean()) / (alloc_costs.std() + 1e-6)
    
    # Concatenate features
    pyg_data['task'].x = torch.cat([pyg_data['task'].x, alloc_times, alloc_costs], dim=1)
    
    # Forward pass
    pred_norm = gnn_aval(pyg_data.x_dict, pyg_data.edge_index_dict, pyg_data.edge_attr_dict)
    pred_real = target_normalizer.denormalize(pred_norm)
    
    print(f"Real: Makespan={real_makespan:.2f}, Cost={real_cost:.2f}")
    print(f"Predicted: Makespan={pred_real[0,0].item():.2f}, Cost={pred_real[0,1].item():.2f}")
    print(f"Error: Makespan={abs(pred_real[0,0].item() - real_makespan):.2f} ({abs((pred_real[0,0].item() - real_makespan)/real_makespan*100):.1f}%)")

print("\nProceeding to GNN_Aloc training...")

# %%
# ==============================================================================
# CÉLULA 7: Modelo 2 - GNN de Alocação (GNN_Aloc)
# ==============================================================================
#
# Esta GNN aprende a PROPOR uma alocação.
#
# Entrada: Um grafo PyG do problema (sem infos de alocação).
# Saída: Para cada nó 'task', um vetor de probabilidades (softmax)
#        sobre qual 'vm' escolher.
#

class GNN_Aloc(torch.nn.Module):
    def __init__(self, hidden_dim, num_vms, num_layers=3, heads = 4):
        super().__init__()

        # A CORREÇÃO ESTÁ AQUI:
        # A GNN_Aloc recebe apenas as 6 features base, e não as 8 (6+2) que a GNN_Aval recebe.
        self.task_emb = nn.Linear(6, hidden_dim) # 6 = dim das features base (mean, std, min para tempo/custo)
        self.vm_emb = nn.Linear(4, hidden_dim)   # 4 = dim inicial

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers): # Número de camadas configurável
            conv = HeteroConv({
                ('task', 'depends_on', 'task'): GATConv(-1 , hidden_dim // heads, heads=heads),
                ('task', 'rev_depends_on', 'task'): GATConv(-1, hidden_dim // heads, heads=heads),
                # Aresta de alocação (com atributos de tempo/custo)
                # APLICA A MESMA CORREÇÃO AQUI
                ('task', 'can_run_on', 'vm'): GATConv((-1, -1), hidden_dim//heads, edge_dim=2, heads=heads, add_self_loops=False),
                ('vm', 'rev_can_run_on', 'task'): GATConv((-1, -1), hidden_dim // heads, edge_dim=2, heads=heads, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        # MLP de Saída (para cada nó 'task', prediz um score por VM)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vms) # Saída: [N_tasks, N_vms] scores
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, gumbel_tau=1.0):
        # 1. Aplicar embeddings de entrada
        x_dict['task'] = self.task_emb(x_dict['task']).relu()
        x_dict['vm'] = self.vm_emb(x_dict['vm']).relu()

        # 2. Camadas de Convolução
        for conv in self.convs:
            # A CORREÇÃO ESTÁ AQUI:
            # Simplesmente passamos todos os dicionários. A HeteroConv
            # é inteligente o suficiente para rotear os 'edge_attr'
            # apenas para as camadas GATConv que os esperam (aquelas com edge_dim=2).
            x_dict_update = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: x_dict[key] + x_dict_update[key].relu() for key in x_dict.keys()}

        # 3. Obter embeddings finais das tarefas
        task_embeddings = x_dict['task']

        # 4. MLP de Saída -> Scores de Alocação
        alloc_logits = self.out_mlp(task_embeddings)

        # 5. Aplicar Gumbel-Softmax
        # MELHORIA: Usar hard=True.
        # Isso usa o "Straight-Through Estimator" (STE).
        # Forward: passa o one-hot (diferente de 'soft_alloc' anterior)
        # Backward: passa o gradiente da versão 'soft'
        hard_alloc = F.gumbel_softmax(alloc_logits, tau=gumbel_tau, hard=True, dim=-1)

        return alloc_logits, hard_alloc

# --- Teste do Modelo (Instanciação) ---
# IMPORTANTE: GNN_Aloc usa apenas as 6 features base (mean, std, min de tempo/custo)
# GNN_Aval usa 8 features (6 base + 2 de alocação: tempo e custo da VM escolhida)
# Por isso precisamos pegar apenas as primeiras 6 colunas

NUM_VMS = 3  # Reduzido
gnn_aloc_model = GNN_Aloc(hidden_dim=HIDDEN_DIM, num_vms=NUM_VMS, num_layers=3, heads=HEADS)
print("\nEstrutura GNN_Aloc:")
print(gnn_aloc_model)

# Teste de forward - USAR APENAS AS 6 PRIMEIRAS FEATURES
# Cria uma cópia dos dados e mantém apenas as 6 features originais
pyg_data_for_aloc = pyg_data.clone()
pyg_data_for_aloc['task'].x = pyg_data['task'].x[:, :6]  # Apenas as 6 features base

logits, hard_alloc = gnn_aloc_model(pyg_data_for_aloc.x_dict, 
                                     pyg_data_for_aloc.edge_index_dict, 
                                     pyg_data_for_aloc.edge_attr_dict)
print(f"\nSaída Logits GNN_Aloc (shape): {logits.shape}") # (N_tasks, N_vms)
print(f"Saída Hard Alloc GNN_Aloc (shape): {hard_alloc.shape}") # (N_tasks, N_vms)
print(f"Exemplo Hard Alloc (task 0): {hard_alloc[0]}") # Deve ser [0., 1., 0.] ou similar
print(f"Soma da Hard Alloc (task 0): {hard_alloc[0].sum().item()}") # Deve ser 1.0

# %%
# ==============================================================================
# CÉLULA 8: Fase 2 - Treinamento da GNN_Aloc (Não-Supervisionado)
# ==============================================================================
#
# O "Loop Mágico" (Artigo 2)
#
# 1. Carregar GNN_Aval pré-treinada e CONGELAR parâmetros
# 2. Carregar o Normalizador
# 3. Gerar um novo Problema (DAG, Matrizes)
# 4. Passar Problema pela GNN_Aloc -> obter 'hard_allocation' (one-hot)
# 5. Calcular features de alocação (tempo/custo) a partir da 'hard_allocation'
# 6. Criar o grafo de entrada para a GNN_Aval (com as features "duras")
# 7. Passar pela GNN_Aval (gradientes ATIVADOS mas parâmetros congelados)
# 8. A SAÍDA da GNN_Aval (normalizada) É A NOSSA LOSS!
# 9. Backpropagate a 'loss' e ATUALIZAR GNN_Aloc.
#

print("\nIniciando Treinamento da GNN_Aloc...")

# 1. Carregar GNN_Aval - CORREÇÃO APLICADA
gnn_aval_frozen = GNN_Aval(hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYERS, heads=HEADS)
gnn_aval_frozen.load_state_dict(torch.load("gnn_aval_pretrained.pth"))

# CORREÇÃO: Congelar parâmetros mas permitir gradientes através do grafo computacional
for param in gnn_aval_frozen.parameters():
    param.requires_grad = False  # Não atualizar os parâmetros durante o treino

# Não usar .eval() para não desativar completamente os gradientes
# Em vez disso, usar um flag para controlar se estamos em modo de avaliação
gnn_aval_frozen.eval()  # Podemos usar .eval() se não tivermos dropout/batch_norm
# Mas precisaremos ativar gradientes temporariamente durante o forward

print("GNN_Aval carregada e parâmetros congelados.")

# 2. Carregar Normalizador
target_normalizer = TargetNormalizer()
try:
    target_normalizer.load_state_dict(torch.load("gnn_aval_normalizer.pth"))
    print("Normalizador de alvos carregado.")
except:
    print("Normalizador não encontrado. Inicializando novo.")
    # Se não existir, crie dados dummy para inicializar
    dummy_y = torch.randn(100, 2) * 100 + 500
    target_normalizer.update(dummy_y)

# 3. Inicializar GNN_Aloc
NUM_VMS = 3  # Reduzido
gnn_aloc = GNN_Aloc(hidden_dim=HIDDEN_DIM, num_vms=NUM_VMS, num_layers=3, heads=HEADS)
optimizer_aloc = optim.Adam(gnn_aloc.parameters(), lr=1e-5) # LR menor é mais seguro

# 4. Adicionar scheduler para learning rate - CORREÇÃO: remover 'verbose'
scheduler_aloc = optim.lr_scheduler.ReduceLROnPlateau(optimizer_aloc, mode='min', 
                                                     factor=0.5, patience=100)

# Pesos da nossa função de utilidade (o que queremos minimizar)
W_TIME = 1.0  # Foco no makespan
W_COST = 0.0  # Custo zero - foco apenas em tempo

# MELHORIA: Annealing do Gumbel-Softmax
gumbel_tau = 2.0 
GUMBEL_TAU_END = 0.5
ANNEAL_RATE = 0.995

# Context manager para ativar gradientes temporariamente
class EnableGradients:
    """Context manager para ativar gradientes em modelos .eval()"""
    def __init__(self, model):
        self.model = model
        self.prev_training = None
        
    def __enter__(self):
        self.prev_training = self.model.training
        self.model.train()  # Temporariamente coloca em modo train para ativar gradientes
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurar o estado original
        if not self.prev_training:
            self.model.eval()

# Listas para armazenar métricas de treino
aloc_losses = []
aloc_makespan_est = []
aloc_cost_est = []
learning_rates = []

# --- Loop de Treinamento GNN_Aloc ---
NUM_EPOCHS_ALOC = 2000  # Aumentado
STEPS_PER_EPOCH_ALOC = 50
BATCH_SIZE_ALOC = 16  # Treinar em batch

# Variáveis para rastrear o melhor modelo
best_loss = float('inf')
best_model_state = None

print(f"\nConfiguração de Treino:")
print(f"- Épocas: {NUM_EPOCHS_ALOC}")
print(f"- Steps por época: {STEPS_PER_EPOCH_ALOC}")
print(f"- Batch size: {BATCH_SIZE_ALOC}")
print(f"- Tau inicial: {gumbel_tau}, Tau final: {GUMBEL_TAU_END}")
print(f"- LR inicial: {optimizer_aloc.param_groups[0]['lr']}")

for epoch in tqdm(range(NUM_EPOCHS_ALOC)):
    # Atualiza Tau (Annealing)
    gumbel_tau = max(GUMBEL_TAU_END, gumbel_tau * ANNEAL_RATE)

    epoch_loss = 0.0
    epoch_makespan = 0.0
    epoch_cost = 0.0
    
    for step in range(STEPS_PER_EPOCH_ALOC):
        optimizer_aloc.zero_grad()
        
        step_loss = 0
        step_makespan = 0
        step_cost = 0
        
        # Processar batch
        for _ in range(BATCH_SIZE_ALOC):
            # 3. Gerar Problema
            num_tasks = 10 + random.randint(-2, 3)  # Reduzido
            dag = generate_workflow_dag(num_tasks)
            num_nodes = dag.number_of_nodes()
            time_m, cost_m = generate_cost_time_matrix(num_nodes, NUM_VMS)

            # Zerar source/sink
            source_node = list(nx.topological_sort(dag))[0]
            sink_node = list(nx.topological_sort(dag))[-1]
            time_m[source_node, :] = 0
            time_m[sink_node, :] = 0
            cost_m[source_node, :] = 0
            cost_m[sink_node, :] = 0

            time_m_t = torch.from_numpy(time_m).float()
            cost_m_t = torch.from_numpy(cost_m).float()

            # Criar grafo base (sem features de alocação)
            data_aloc = create_pyg_data(dag, time_m, cost_m)

            # 4. Forward GNN_Aloc -> 'hard_alloc' (one-hot)
            _, hard_alloc = gnn_aloc(data_aloc.x_dict,
                                    data_aloc.edge_index_dict,
                                    data_aloc.edge_attr_dict,
                                    gumbel_tau=gumbel_tau)

            # 5. Calcular features de alocação
            # Multiplicar one-hot pelas matrizes de tempo/custo
            alloc_time = (hard_alloc * time_m_t).sum(dim=1, keepdim=True)
            alloc_cost = (hard_alloc * cost_m_t).sum(dim=1, keepdim=True)

            # Normalizar (para a entrada da GNN_Aval)
            alloc_time_norm = (alloc_time - alloc_time.mean()) / (alloc_time.std() + 1e-6)
            alloc_cost_norm = (alloc_cost - alloc_cost.mean()) / (alloc_cost.std() + 1e-6)

            # 6. Criar grafo de entrada para GNN_Aval
            data_aval_in = data_aloc.clone()
            # Concatenar features base + features de alocação
            data_aval_in['task'].x = torch.cat([data_aloc['task'].x, alloc_time_norm, alloc_cost_norm], dim=1)

            # 7. Forward GNN_Aval - CORREÇÃO CHAVE: ATIVAR GRADIENTES
            # Usar context manager para ativar gradientes temporariamente
            with EnableGradients(gnn_aval_frozen):
                pred_metrics_norm = gnn_aval_frozen(data_aval_in.x_dict,
                                                   data_aval_in.edge_index_dict,
                                                   data_aval_in.edge_attr_dict)

            pred_makespan_norm = pred_metrics_norm[0, 0]
            pred_cost_norm = pred_metrics_norm[0, 1]

            # 8. Calcular a Loss (combinação linear das métricas normalizadas)
            loss = W_TIME * pred_makespan_norm + W_COST * pred_cost_norm

            # Acumular gradientes - IMPORTANTE: manter no escopo do EnableGradients
            loss.backward()
            
            step_loss += loss.item()
            
            # Para métricas (sem gradiente)
            with torch.no_grad():
                pred_denorm = target_normalizer.denormalize(pred_metrics_norm.detach())
                step_makespan += pred_denorm[0, 0].item()
                step_cost += pred_denorm[0, 1].item()

        # Clip gradientes para evitar exploding gradients
        torch.nn.utils.clip_grad_norm_(gnn_aloc.parameters(), max_norm=1.0)
        
        # 9. Atualizar GNN_Aloc
        optimizer_aloc.step()

        epoch_loss += step_loss / BATCH_SIZE_ALOC
        epoch_makespan += step_makespan / BATCH_SIZE_ALOC
        epoch_cost += step_cost / BATCH_SIZE_ALOC

    # Média por step
    avg_epoch_loss = epoch_loss / STEPS_PER_EPOCH_ALOC
    avg_epoch_makespan = epoch_makespan / STEPS_PER_EPOCH_ALOC
    avg_epoch_cost = epoch_cost / STEPS_PER_EPOCH_ALOC
    
    # Armazenar métricas
    aloc_losses.append(avg_epoch_loss)
    aloc_makespan_est.append(avg_epoch_makespan)
    aloc_cost_est.append(avg_epoch_cost)
    learning_rates.append(optimizer_aloc.param_groups[0]['lr'])
    
    # Atualizar scheduler
    scheduler_aloc.step(avg_epoch_loss)
    
    # Salvar melhor modelo
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_state = gnn_aloc.state_dict().copy()
        torch.save(best_model_state, "gnn_aloc_best.pth")

    if (epoch + 1) % 50 == 0:
        # Denormalizar loss para interpretação
        with torch.no_grad():
            # Criar um tensor dummy para denormalizar
            loss_tensor = torch.tensor([[avg_epoch_loss, avg_epoch_cost]], dtype=torch.float32)
            # Apenas para referência - não é exatamente correto mas dá uma ideia
            makespan_approx = avg_epoch_makespan
        
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS_ALOC}]")
        print(f"  Tau: {gumbel_tau:.3f}, LR: {learning_rates[-1]:.2e}")
        print(f"  Loss (Norm): {avg_epoch_loss:.4f}")
        print(f"  Makespan (Est.): {avg_epoch_makespan:.2f}")
        print(f"  Custo (Est.): {avg_epoch_cost:.2f}")
        
        # Validação rápida a cada 100 épocas
        if (epoch + 1) % 100 == 0:
            print("  [Validação] Testando alocação em 5 problemas...")
            val_makespans = []
            val_costs = []
            
            for _ in range(5):
                # Gerar problema de validação
                num_tasks = 12
                dag = generate_workflow_dag(num_tasks)
                num_nodes = dag.number_of_nodes()
                time_m, cost_m = generate_cost_time_matrix(num_nodes, NUM_VMS)
                
                source_node = list(nx.topological_sort(dag))[0]
                sink_node = list(nx.topological_sort(dag))[-1]
                time_m[source_node, :] = 0
                time_m[sink_node, :] = 0
                cost_m[source_node, :] = 0
                cost_m[sink_node, :] = 0
                
                data_val = create_pyg_data(dag, time_m, cost_m)
                
                with torch.no_grad():
                    logits_val, _ = gnn_aloc(data_val.x_dict,
                                           data_val.edge_index_dict,
                                           data_val.edge_attr_dict,
                                           gumbel_tau=0.1)  # Tau baixo para decisão mais certa
                    
                    hard_alloc_val = logits_val.argmax(dim=-1).cpu().numpy()
                    alloc_dict_val = {node_id: vm_id for node_id, vm_id in enumerate(hard_alloc_val)}
                    
                    m_val, c_val = calculate_metrics(dag, alloc_dict_val, time_m, cost_m)
                    val_makespans.append(m_val)
                    val_costs.append(c_val)
            
            avg_val_makespan = np.mean(val_makespans)
            avg_val_cost = np.mean(val_costs)
            print(f"    Makespan médio: {avg_val_makespan:.2f}, Custo médio: {avg_val_cost:.2f}")

print("\nTreinamento da GNN_Aloc concluído.")

# Carregar melhor modelo
if best_model_state is not None:
    gnn_aloc.load_state_dict(best_model_state)
    print(f"Melhor modelo carregado (loss: {best_loss:.4f})")

# Salvar modelo final
torch.save(gnn_aloc.state_dict(), "gnn_aloc_final.pth")
print("Modelo final salvo.")

# Plot dos gráficos de treino da GNN_Aloc
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Loss de treino
axes[0, 0].plot(aloc_losses)
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Loss (Normalizada)')
axes[0, 0].set_title('Loss de Treino da GNN_Aloc')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=best_loss, color='r', linestyle='--', alpha=0.7, label=f'Melhor: {best_loss:.4f}')
axes[0, 0].legend()

# Gráfico 2: Makespan estimado
axes[0, 1].plot(aloc_makespan_est)
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Makespan Estimado')
axes[0, 1].set_title('Makespan Estimado (Treino)')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Custo estimado
axes[1, 0].plot(aloc_cost_est)
axes[1, 0].set_xlabel('Época')
axes[1, 0].set_ylabel('Custo Estimado')
axes[1, 0].set_title('Custo Estimado (Treino)')
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Learning Rate
axes[1, 1].plot(learning_rates)
axes[1, 1].set_xlabel('Época')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("treino_gnn_aloc.png")  # Salva em arquivo
plt.close()  # Fecha a janela para não bloquear

# Plot adicional: Loss vs Makespan
fig, ax1 = plt.subplots(figsize=(12, 5))

color1 = 'tab:red'
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss (Normalizada)', color=color1)
ax1.plot(aloc_losses, color=color1, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Makespan Estimado', color=color2)
ax2.plot(aloc_makespan_est, color=color2, alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Correlação entre Loss e Makespan Estimado')
fig.tight_layout()
plt.savefig("correlacao_loss_makespan.png")
plt.close()

print("Gráficos salvos em arquivos PNG.")

# %%
# ==============================================================================
# CÉLULA DE TESTE: Verificação do Pipeline
# ==============================================================================
#

print("\n" + "="*60)
print("TESTE RÁPIDO DO PIPELINE COMPLETO")
print("="*60)

# Testar com um problema pequeno
test_dag = generate_workflow_dag(6)
num_nodes = test_dag.number_of_nodes()
test_times, test_costs = generate_cost_time_matrix(num_nodes, NUM_VMS)

# Zerar source/sink
source_node = list(nx.topological_sort(test_dag))[0]
sink_node = list(nx.topological_sort(test_dag))[-1]
test_times[source_node, :] = 0
test_times[sink_node, :] = 0
test_costs[source_node, :] = 0
test_costs[sink_node, :] = 0

# Criar dados PyG
data_test = create_pyg_data(test_dag, test_times, test_costs)

# Testar GNN_Aloc
print("\n1. Testando GNN_Aloc...")
with torch.no_grad():
    logits, hard_alloc = gnn_aloc(data_test.x_dict, 
                                 data_test.edge_index_dict,
                                 data_test.edge_attr_dict)
    
    hard_alloc_np = hard_alloc.argmax(dim=-1).cpu().numpy()
    alloc_dict = {node_id: vm_id for node_id, vm_id in enumerate(hard_alloc_np)}
    
    print(f"   Alocação gerada: {alloc_dict}")
    
    # Calcular métricas reais
    real_makespan, real_cost = calculate_metrics(test_dag, alloc_dict, test_times, test_costs)
    print(f"   Makespan real: {real_makespan:.2f}, Custo real: {real_cost:.2f}")

# Testar pipeline completo (GNN_Aloc + GNN_Aval)
print("\n2. Testando Pipeline Completo...")

# Calcular features de alocação
time_m_t = torch.from_numpy(test_times).float()
cost_m_t = torch.from_numpy(test_costs).float()
alloc_time = (hard_alloc * time_m_t).sum(dim=1, keepdim=True)
alloc_cost = (hard_alloc * cost_m_t).sum(dim=1, keepdim=True)

# Normalizar
alloc_time_norm = (alloc_time - alloc_time.mean()) / (alloc_time.std() + 1e-6)
alloc_cost_norm = (alloc_cost - alloc_cost.mean()) / (alloc_cost.std() + 1e-6)

# Preparar entrada para GNN_Aval
data_aval_in = data_test.clone()
data_aval_in['task'].x = torch.cat([data_test['task'].x, alloc_time_norm, alloc_cost_norm], dim=1)

# Forward GNN_Aval
with EnableGradients(gnn_aval_frozen):
    pred_norm = gnn_aval_frozen(data_aval_in.x_dict,
                               data_aval_in.edge_index_dict,
                               data_aval_in.edge_attr_dict)
    
    pred_denorm = target_normalizer.denormalize(pred_norm)
    pred_makespan = pred_denorm[0, 0].item()
    pred_cost = pred_denorm[0, 1].item()
    
    print(f"   GNN_Aval prediz: Makespan = {pred_makespan:.2f}, Custo = {pred_cost:.2f}")
    print(f"   Erro: Makespan {abs(pred_makespan - real_makespan):.2f} ({abs((pred_makespan - real_makespan)/real_makespan*100):.1f}%)")

print("\n" + "="*60)
print("TESTE CONCLUÍDO - PRONTO PARA AVALIAÇÃO FINAL")
print("="*60)

# %%
# ==============================================================================
# CÉLULA 9: Avaliação e Análise
# ==============================================================================
#
# Agora comparamos nossa GNN_Aloc treinada com uma heurística simples.
#
# 1. Carregar GNN_Aloc treinada.
# 2. Gerar um novo conjunto de problemas (teste).
# 3. Para cada problema:
#    a. Calcular alocação da GNN_Aloc (agora usando argmax, não gumbel)
#    b. Calcular alocação da Heurística (e.g., "GRASP" ou "Mais Rápido")
# 4. Usar o SIMULADOR (Célula 3) para obter os CUSTOS REAIS de ambas.
# 5. Comparar os resultados.
#

print("\nIniciando Avaliação...")

# 1. Carregar modelo
gnn_aloc_eval = GNN_Aloc(hidden_dim=HIDDEN_DIM, num_vms=NUM_VMS, num_layers=3, heads=HEADS)
gnn_aloc_eval.load_state_dict(torch.load("gnn_aloc_final.pth"))
gnn_aloc_eval.eval()

# 2. Heurísticas
def heuristic_fastest(dag, time_m):
    alloc_dict = {}
    for node_id in dag.nodes():
        vm_id = np.argmin(time_m[node_id])
        alloc_dict[node_id] = vm_id
    return alloc_dict

def heuristic_cheapest(dag, cost_m):
    alloc_dict = {}
    for node_id in dag.nodes():
        vm_id = np.argmin(cost_m[node_id])
        alloc_dict[node_id] = vm_id
    return alloc_dict

# --- Loop de Avaliação ---
NUM_TEST_PROBLEMS = 100
results = {
    'gnn_makespan': [], 'gnn_cost': [],
    'fastest_makespan': [], 'fastest_cost': [],
    'cheapest_makespan': [], 'cheapest_cost': [],
    'random_makespan': [], 'random_cost': [],
}

for i in tqdm(range(NUM_TEST_PROBLEMS)):
    # Gerar problema de teste
    num_tasks = 12  # Tamanho fixo para teste
    dag = generate_workflow_dag(num_tasks)
    num_nodes = dag.number_of_nodes()
    time_m, cost_m = generate_cost_time_matrix(num_nodes, NUM_VMS)

    source_node = list(nx.topological_sort(dag))[0]
    sink_node = list(nx.topological_sort(dag))[-1]
    time_m[source_node, :] = 0
    time_m[sink_node, :] = 0
    cost_m[source_node, :] = 0
    cost_m[sink_node, :] = 0

    # --- a. Avaliação da GNN ---
    data_test = create_pyg_data(dag, time_m, cost_m)
    with torch.no_grad():
        logits_gnn, _ = gnn_aloc_eval(data_test.x_dict,
                                     data_test.edge_index_dict,
                                     data_test.edge_attr_dict)

    # Usar argmax para obter a alocação "hard" (decisão final)
    hard_alloc_gnn = logits_gnn.argmax(dim=-1).cpu().numpy()
    alloc_dict_gnn = {node_id: vm_id for node_id, vm_id in enumerate(hard_alloc_gnn)}

    m_gnn, c_gnn = calculate_metrics(dag, alloc_dict_gnn, time_m, cost_m)
    results['gnn_makespan'].append(m_gnn)
    results['gnn_cost'].append(c_gnn)

    # --- b. Avaliação da Heurística "Fastest" ---
    alloc_dict_fastest = heuristic_fastest(dag, time_m)
    m_fast, c_fast = calculate_metrics(dag, alloc_dict_fastest, time_m, cost_m)
    results['fastest_makespan'].append(m_fast)
    results['fastest_cost'].append(c_fast)

    # --- c. Avaliação da Heurística "Cheapest" ---
    alloc_dict_cheapest = heuristic_cheapest(dag, cost_m)
    m_cheap, c_cheap = calculate_metrics(dag, alloc_dict_cheapest, time_m, cost_m)
    results['cheapest_makespan'].append(m_cheap)
    results['cheapest_cost'].append(c_cheap)
    
    # --- d. Avaliação da Heurística "Random" ---
    alloc_dict_random = heuristic_random(dag, NUM_VMS)
    m_rand, c_rand = calculate_metrics(dag, alloc_dict_random, time_m, cost_m)
    results['random_makespan'].append(m_rand)
    results['random_cost'].append(c_rand)

# --- 5. Comparar Resultados ---
print("\n" + "="*60)
print("RESULTADOS DA AVALIAÇÃO (Média de 100 execuções)")
print("="*60)

print(f"\nModelo GNN (Treinado para Makespan):")
print(f"  - Makespan Médio: {np.mean(results['gnn_makespan']):.2f} ± {np.std(results['gnn_makespan']):.2f}")
print(f"  - Custo Médio:    {np.mean(results['gnn_cost']):.2f} ± {np.std(results['gnn_cost']):.2f}")

print(f"\nHeurística 'Mais Rápido':")
print(f"  - Makespan Médio: {np.mean(results['fastest_makespan']):.2f} ± {np.std(results['fastest_makespan']):.2f}")
print(f"  - Custo Médio:    {np.mean(results['fastest_cost']):.2f} ± {np.std(results['fastest_cost']):.2f}")

print(f"\nHeurística 'Mais Barato':")
print(f"  - Makespan Médio: {np.mean(results['cheapest_makespan']):.2f} ± {np.std(results['cheapest_makespan']):.2f}")
print(f"  - Custo Médio:    {np.mean(results['cheapest_cost']):.2f} ± {np.std(results['cheapest_cost']):.2f}")

print(f"\nHeurística 'Aleatória':")
print(f"  - Makespan Médio: {np.mean(results['random_makespan']):.2f} ± {np.std(results['random_makespan']):.2f}")
print(f"  - Custo Médio:    {np.mean(results['random_cost']):.2f} ± {np.std(results['random_cost']):.2f}")

# Calcular melhorias percentuais
gnn_mean_makespan = np.mean(results['gnn_makespan'])
fast_mean_makespan = np.mean(results['fastest_makespan'])
cheap_mean_makespan = np.mean(results['cheapest_makespan'])
rand_mean_makespan = np.mean(results['random_makespan'])

improvement_vs_fastest = (fast_mean_makespan - gnn_mean_makespan) / fast_mean_makespan * 100
improvement_vs_cheapest = (cheap_mean_makespan - gnn_mean_makespan) / cheap_mean_makespan * 100
improvement_vs_random = (rand_mean_makespan - gnn_mean_makespan) / rand_mean_makespan * 100

print(f"\n{'='*60}")
print("MELHORIAS PERCENTUAIS (Makespan)")
print(f"{'='*60}")
print(f"GNN vs Mais Rápido: {improvement_vs_fastest:+.1f}%")
print(f"GNN vs Mais Barato: {improvement_vs_cheapest:+.1f}%")
print(f"GNN vs Aleatório:   {improvement_vs_random:+.1f}%")

# Plot dos resultados comparativos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot do Makespan
makespan_data = [results['gnn_makespan'], results['fastest_makespan'], 
                 results['cheapest_makespan'], results['random_makespan']]
axes[0].boxplot(makespan_data, labels=['GNN', 'Mais Rápido', 'Mais Barato', 'Aleatório'])
axes[0].set_ylabel('Makespan')
axes[0].set_title('Comparação de Makespan')
axes[0].grid(True, alpha=0.3)

# Boxplot do Custo
cost_data = [results['gnn_cost'], results['fastest_cost'], 
             results['cheapest_cost'], results['random_cost']]
axes[1].boxplot(cost_data, labels=['GNN', 'Mais Rápido', 'Mais Barato', 'Aleatório'])
axes[1].set_ylabel('Custo Total')
axes[1].set_title('Comparação de Custo')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Scatter plot: Makespan vs Custo
plt.figure(figsize=(10, 6))
plt.scatter(results['gnn_makespan'], results['gnn_cost'], alpha=0.6, label='GNN', color='blue')
plt.scatter(results['fastest_makespan'], results['fastest_cost'], alpha=0.6, label='Mais Rápido', color='red')
plt.scatter(results['cheapest_makespan'], results['cheapest_cost'], alpha=0.6, label='Mais Barato', color='green')
plt.scatter(results['random_makespan'], results['random_cost'], alpha=0.6, label='Aleatório', color='orange')

plt.xlabel('Makespan')
plt.ylabel('Custo Total')
plt.title('Trade-off Makespan vs Custo (100 problemas)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

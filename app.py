import streamlit as st

# Function to display code examples
def display_code(code: str):
    st.code(code, language='python')

# Code examples
BFS_code = '''
print("22EC139 SANJAY S")
print("Experiment 4 - BFS for Pac-Man Navigation\n")
from collections import deque
maze = [
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [2, 0, 0, 0],
    [0, 1, 1, 0]
]
start = (0, 0)
goal = (3, 3)
moves = [(-1,0), (1,0), (0,-1), (0,1)]
def bfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    visited = set()
    queue = deque([(start, [start])])
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                maze[nx][ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    return None
path = bfs(maze, start, goal)
print("Maze Path:", path if path else "No path found.")
'''

A_Star = '''
print("22EC139 SANJAY S")
print("Experiment 5 - A* Search Algorithm\n")
import heapq
grid = [
    [0, 0, 1],
    [0, 9, 0],
    [2, 0, 0]
]
costs = {0: 1, 1: 3, 2: 2, 9: float('inf')}
start, goal = (0, 0), (2, 2)
def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def a_star(grid, start, goal):
    q = [(0 + h(start, goal), 0, start, [start])]
    visited = set()
    while q:
        _, cost, (x, y), path = heapq.heappop(q)
        if (x, y) == goal: return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<3 and 0<=ny<3 and (nx,ny) not in visited:
                c = costs[grid[nx][ny]]
                if c < float('inf'):
                    heapq.heappush(q, (cost+c+h((nx,ny), goal), cost+c, (nx,ny), path+[(nx,ny)]))
                    visited.add((nx,ny))
    return None
print("Path:", a_star(grid, start, goal))
'''

Alpha_Beta = '''
print("22EC139 SANJAY S")
print("Experiment 6 - Alpha-Beta Pruning (Tic-Tac-Toe)\n")
def winner(b):
    w = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for i,j,k in w:
        if b[i]==b[j]==b[k] and b[i]!=' ': return b[i]
    return 'D' if ' ' not in b else None
def minimax(b, maxp, a, bta):
    win = winner(b)
    if win: return {'X':1, 'O':-1, 'D':0}[win]
    vals = []
    for i in range(9):
        if b[i]==' ':
            b[i] = 'X' if maxp else 'O'
            val = minimax(b, not maxp, a, bta)
            b[i] = ' '
            vals.append(val)
            if maxp:
                a = max(a,val)
                if a >= bta: break
            else:
                bta = min(bta,val)
                if bta <= a: break
    return max(vals) if maxp else min(vals)
def best_move(b):
    best, move = -2, -1
    for i in range(9):
        if b[i]==' ':
            b[i]='X'
            val = minimax(b, False, -2, 2)
            b[i]=' '
            if val > best: best, move = val, i
    return move
board = ['X','O','X',' ','O',' ',' ',' ',' ']
print("Best move for X is:", best_move(board))
'''

ID_3 = '''
print("22EC139 SANJAY S")
print("Experiment 7 - ID3 Decision Tree\n")
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
features = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain']
humidity = ['High', 'High', 'High', 'High', 'Normal']
play = ['No', 'No', 'Yes', 'Yes', 'Yes']
X = list(zip(features, humidity))
le_f = LabelEncoder()
le_h = LabelEncoder()
le_y = LabelEncoder()
X_enc = list(zip(le_f.fit_transform(features), le_h.fit_transform(humidity)))
y_enc = le_y.fit_transform(play)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_enc, y_enc)
sample = [(le_f.transform(['Sunny'])[0], le_h.transform(['Normal'])[0])]
pred = clf.predict(sample)
print("Prediction for Sunny & Normal:", le_y.inverse_transform(pred)[0])
'''

ANN_Code= '''
print("22EC139 SANJAY S")
print("Experiment 8 - ANN using Backpropagation\n")
from sklearn.neural_network import MLPClassifier
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]
clf = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', max_iter=1000)
clf.fit(X, y)
out = clf.predict([[0,0], [0,1], [1,1]])
print("Predictions:", out)
print("Accuracy:", clf.score(X, y))
'''

Naiv_accu = '''
print("22EC139 SANJAY S exp_9\n")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("=== Naive Bayes Classifier Results ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
'''

acccu_press = '''
print("22EC139 SANJAY  exp_10\n")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
data = {
    'text': [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "U dun say so early hor... U c already then say...",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
        "I'm gonna be home soon and I don't want to talk about this stuff anymore tonight",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.",
        "I'm back, lemme know when you're free",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "I’ve been searching for the right words to thank you for this breather.",
        "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message!"
    ],
    'label': ['spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
print("=== Naive Bayes Document Classifier Results ===")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
'''

heart_disease = '''
print("22EC139 SANJAY exp_11\n")
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
model = DiscreteBayesianNetwork([
    ('Age', 'HeartDisease'),
    ('Exercise', 'HeartDisease'),
    ('Cholesterol', 'HeartDisease')
])
cpd_age = TabularCPD(variable='Age', variable_card=2, values=[[0.6], [0.4]])  # 0: Young, 1: Old
cpd_exercise = TabularCPD(variable='Exercise', variable_card=2, values=[[0.7], [0.3]])  # 0: Yes, 1: No
cpd_chol = TabularCPD(variable='Cholesterol', variable_card=2, values=[[0.8], [0.2]])  # 0: Normal, 1: High
cpd_heart = TabularCPD(
    variable='HeartDisease', variable_card=2,
    values=[
        [0.9, 0.8, 0.7, 0.6, 0.7, 0.5, 0.4, 0.3],  # No Disease
        [0.1, 0.2, 0.3, 0.4, 0.3, 0.5, 0.6, 0.7]   # Disease
    ],
    evidence=['Age', 'Exercise', 'Cholesterol'],
    evidence_card=[2, 2, 2]
)
model.add_cpds(cpd_age, cpd_exercise, cpd_chol, cpd_heart)
infer = VariableElimination(model)
result = infer.query(variables=['HeartDisease'], evidence={'Age': 1, 'Exercise': 1, 'Cholesterol': 1})
print("=== Bayesian Network Inference Result ===")
print(result)
plt.figure(figsize=(6, 4))
G = nx.DiGraph()
G.add_edges_from(model.edges())
nx.draw(
    G, with_labels=True, node_size=2000, node_color="lightblue",
    font_size=12, font_weight="bold", arrows=True, arrowsize=20
)
plt.title("Bayesian Network - Heart Disease Diagnosis")
plt.show()
'''

knn_code = '''
print("22EC139 SANJAY exp_12\n")
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
data = load_iris()
X = data.data  # Only features
kmeans = KMeans(n_clusters=3, random_state=22)
kmeans_labels = kmeans.fit_predict(X)
kmeans_score = silhouette_score(X, kmeans_labels)
em = GaussianMixture(n_components=3, random_state=22)
em_labels = em.fit_predict(X)
em_score = silhouette_score(X, em_labels)
print("=== Clustering Comparison (Roll No: 22EC304) ===")
print(f"K-Means Silhouette Score: {kmeans_score:.4f}")
print(f"EM (GMM) Silhouette Score: {em_score:.4f}")
if kmeans_score > em_score:
    print("Result: K-Means performed better based on silhouette score.")
elif em_score > kmeans_score:
    print("Result: EM (Gaussian Mixture) performed better based on silhouette score.")
else:
    print("Result: Both algorithms performed equally.")
'''

# Streamlit App Layout
st.title("ML Practical")

# Sidebar for selecting experiments
selected_experiment = st.sidebar.selectbox(
    "Select an Experiment", 
    ["BFS_code", "A_Star", "Alpha_Beta", 
     "ID_3", "ANN_Code", "Naiv_accu","acccu_press","heart_disease ","knn_code"]
)

# Display corresponding code
if selected_experiment == "BFS_code":
    st.header("BFS_code")
    display_code(BFS_code)
elif selected_experiment == "A_Star":
    st.header("A_Star")
    display_code(A_Star)
elif selected_experiment == "Alpha_Beta":
    st.header("Alpha_Beta")
    display_code(Alpha_Beta)
elif selected_experiment == "ID_3":
    st.header("ID_3")
    display_code(ID_3)
elif selected_experiment == "ANN_Code":
    st.header("ANN_Code")
    display_code(ANN_Code)
elif selected_experiment == "Naiv_accu":
    st.header("Naiv_accu")
    display_code(Naiv_accu)
elif selected_experiment == "acccu_press":
    st.header("acccu_press")
    display_code(acccu_press)
elif selected_experiment == "heart_disease":
    st.header("heart_disease")
    display_code(heart_disease)
elif selected_experiment == "knn_code":
    st.header("knn_code")
    display_code(knn_code)

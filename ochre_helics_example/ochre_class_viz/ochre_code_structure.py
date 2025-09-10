import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from ochre import Equipment  # adjust import

def build_inheritance_graph(base_class):
    G = nx.DiGraph()
    def add_subclasses(cls):
        for sub in cls.__subclasses__():
            G.add_edge(cls.__name__, sub.__name__)
            add_subclasses(sub)
    add_subclasses(base_class)
    return G


G = build_inheritance_graph(Equipment)

pos = graphviz_layout(G, prog="dot")

plt.figure(figsize=(24, 10))
nx.draw(G, pos,
        with_labels=True,
        node_size=3000,
        node_color="#AED6F1",
        edgecolors="black",
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=20)
plt.title("Equipment Inheritance Tree", fontsize=14, fontweight="bold")
plt.tight_layout()
# plt.savefig("equipment_inheritance.png", dpi=300, bbox_inches="tight")
plt.show()
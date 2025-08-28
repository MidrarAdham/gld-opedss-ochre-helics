# import inspect
# import networkx as nx
# import matplotlib.pyplot as plt
# import ochre  # assuming your package is called ochre

# # Choose the base class
# from ochre import Equipment  # adjust import path

# def build_inheritance_graph(base_class):
#     G = nx.DiGraph()

#     def add_subclasses(cls):
#         for sub in cls.__subclasses__():
#             G.add_edge(cls.__name__, sub.__name__)
#             add_subclasses(sub)

#     add_subclasses(base_class)
#     return G

# # Build graph starting at Equipment
# G = build_inheritance_graph(Equipment)

# # Draw with NetworkX
# plt.figure(figsize=(10, 6))
# pos = nx.spring_layout(G, k=0.5, iterations=50)
# # nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10, arrows=True)
# nx.draw(
#     G, pos,
#     with_labels=True,
#     node_size=4000,
#     node_color="#AED6F1",   # soft blue
#     edgecolors="black",     # border around nodes
#     font_size=10,
#     font_weight="bold",
#     arrows=True,
#     arrowsize=20,
#     linewidths=1.5
# )

# plt.title("Inheritance Graph from Equipment")
# plt.show()

# import networkx as nx
# import matplotlib.pyplot as plt
# from ochre import Equipment  # adjust import

# def build_inheritance_graph(base_class):
#     G = nx.DiGraph()
#     def add_subclasses(cls):
#         for sub in cls.__subclasses__():
#             G.add_edge(cls.__name__, sub.__name__)
#             add_subclasses(sub)
#     add_subclasses(base_class)
#     return G

# def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
#     def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, xcenter, pos=None, parent=None):
#         if pos is None: pos = {root: (xcenter, vert_loc)}
#         else: pos[root] = (xcenter, vert_loc)
#         children = list(G.successors(root))
#         if children:
#             dx = width / len(children)
#             nextx = xcenter - width/2 - dx/2
#             for child in children:
#                 nextx += dx
#                 pos = _hierarchy_pos(G, child, leftmost, dx, vert_gap, vert_loc-vert_gap, nextx, pos, root)
#         return pos
#     return _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, xcenter)

# # Build tree
# G = build_inheritance_graph(Equipment)

# # Layout
# pos = hierarchy_pos(G, "Equipment")

# # Draw edges
# nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20, width=1.5)

# # Draw nodes with styles
# nx.draw_networkx_nodes(G, pos, nodelist=["Equipment"], node_shape="s", node_color="#F1948A", node_size=4500, edgecolors="black")
# others = [n for n in G.nodes if n != "Equipment"]
# nx.draw_networkx_nodes(G, pos, nodelist=others, node_shape="o", node_color="#85C1E9", node_size=3500, edgecolors="black")

# # Labels
# nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

# plt.title("Equipment Inheritance Tree", fontsize=14, fontweight="bold")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("equipment_inheritance.png", dpi=300, bbox_inches="tight")
# plt.show()

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

# Build
G = build_inheritance_graph(Equipment)

# Use Graphviz "dot" layout
pos = graphviz_layout(G, prog="dot")

# Draw nicely
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
plt.savefig("equipment_inheritance.png", dpi=300, bbox_inches="tight")
plt.show()


# IEEE 9500-Node Distribution Network — Interactive Visualization

## Overview

`ieee9500_v4.html` is a self-contained, browser-based interactive visualization of the IEEE 9500-Node Distribution Test System. It requires no internet connection, no installation, and no server — open it directly in any modern browser (Firefox, Chrome, or Edge).

The network is drawn using real geographic coordinates derived from `model_busxy.dss`, placing every node at its actual longitude and latitude in the Snoqualmie, WA area (~119°W, 46.7°N). The layout therefore reflects the true spatial relationships between equipment in the model, and closely matches the feeder plots shown in PNNL report PNNL-33471.

The full network contains **5,294 nodes** and **5,322 edges**, parsed from `model_base.dss` and supplemented with connectivity from `model_dict.json`.

---

## Layout

The interface is divided into two regions:

- **Left sidebar** — all controls and information panels
- **Main canvas** — the interactive network map

---

## Navigation

### Pan and Zoom

| Action | Result |
|---|---|
| Scroll wheel | Zoom in / out |
| Click and drag on empty canvas | Pan the view |
| Click a node or edge | Select it and show its properties |
| Click empty canvas | Deselect and clear highlights |

### Toolbar buttons (top-right of canvas)

| Button | Action |
|---|---|
| ⊡ | Fit the entire network into the current window |
| + | Zoom in |
| − | Zoom out |
| ✕ | Clear all selections, highlights, and search results |

---

## Sidebar Panels

### 1. Search Node

Type any part of a node name into the search box to find it on the map.

**How it works:**

- As you type, the visualization scans all node names for ones that contain your query as a substring. For example, searching `m1142` will match `m1142843`, `m1142874`, and any other node whose name contains that string.
- Matching nodes are highlighted in **orange** and everything else is faded out.
- If exactly **one** node matches, the map automatically zooms to it, shows its properties in the Selected Element panel, activates the hop neighborhood highlight (see below), and traces its supply path back to its substation (shown in cyan).
- If **multiple** nodes match, all are highlighted and you can zoom in manually to distinguish them.

**Highlight neighbors — Hops:**

Next to the search box is a numeric field labelled "Highlight neighbors: _N_ hops."

A **hop** is one step along a network edge from one node to an adjacent node. Setting this to 2 means: starting from the found node, show every node reachable by travelling across at most 2 edges.

- **1 hop** — only the directly connected neighbors (e.g. the upstream line and immediate downstream branches)
- **2 hops** — neighbors of neighbors; useful for seeing the local section of feeder
- **5+ hops** — a larger portion of the feeder stretching in both directions from the node

The colors used for hop highlighting are:

| Color | Meaning |
|---|---|
| Yellow | The focus node (the one you searched for or clicked) |
| Orange | 1 hop away (directly connected) |
| Blue | 2 or more hops away (within the hop limit) |
| Amber edges | All edges within the highlighted neighborhood |
| Cyan edges | Supply path from the node back to its substation |
| Faded grey | Everything outside the highlighted neighborhood |

You can change the hop count at any time and re-click a node to re-apply.

---

### 2. Feeder Filter

The IEEE 9500 model has three distribution feeders (S1, S2, S3) supplied by three separate 69kV substations, plus a 69kV sub-transmission network connecting them to the 115kV source bus.

**Feeder color coding:**

| Pill / Color | Feeder | Description |
|---|---|---|
| S1 (blue) | Substation 1 feeder | Upper-right area; typical modern grid with ~10% rooftop PV |
| S2 (green) | Substation 2 feeder | Lower-left area; future smart grid with 100% PV in New Neighborhood |
| S3 (red) | Substation 3 feeder | Upper-left area; legacy system with CHP steam plant |
| 69kV (orange) | Sub-transmission | The 69kV lines and buses connecting the three substations to the source |

**Feeder pills:**

Click a colored pill (S1, S2, S3, or 69kV) to toggle that feeder on or off. A dimmed pill means that feeder is currently deselected.

**Isolate selected feeder:**

Hides all feeders whose pills are currently OFF, leaving only the active ones visible, and fits the view to show only those nodes. This is useful for focusing on a specific feeder when investigating a fault or anomaly.

**Show all feeders:**

Restores all four groups to visible and re-fits the view to show the entire network.

---

### 3. Show / Hide Edge Types

Checkboxes to toggle visibility of each category of network edge (connection). Uncheck a type to hide those connections; re-check to show them again.

| Edge Type | Count | Description |
|---|---|---|
| Distribution lines | 2,634 | The main overhead and underground MV (12.47kV) lines forming the feeder backbone. Colored by feeder (blue/green/red). |
| Triplex lines | 1,275 | Low-voltage (120/240V) secondary service lines running from a service transformer to a customer meter. Usually short (~50ft). Shown in dark purple. |
| Service transformers | 1,275 | The pole-top center-tapped transformers stepping down from 12.47kV (MV) to 120/240V (LV) for each residential customer. Shown in dark green. |
| Sub/reg transformers | 29 | Substation LTC transformers (69/12.47kV) and pole-top voltage regulators. These are the larger power transformers in the model. Shown in amber. |
| Switches | 109 | All switching devices including sectionalizers, reclosers, fuses, and tie switches. Normally-closed switches are shown in teal; **normally-open tie switches** (which define the boundary between feeders) are shown in **red**. |

Hiding triplex lines and service transformers is recommended when investigating the MV network, as it significantly reduces visual clutter at lower zoom levels.

---

### 4. Node Legend

Explains the color and size of node markers on the map.

| Color | Node Type | Description |
|---|---|---|
| Orange (large, outlined) | Source / 115kV bus | The 115kV infinite source bus — the single point of supply for the entire model |
| Yellow (large, outlined) | Substation / 69kV bus | High-side and low-side buses of the three substations and the 69kV sub-transmission nodes |
| Red (medium) | Regulator bus | The intermediate bus node on either side of a voltage regulator transformer |
| Dark orange (medium) | Junction | A node with more than 3 connections; typically a branch point on the main feeder trunk |
| Light green (small) | Triplex secondary | The LV node on the customer side of a service transformer, connected to the triplex service line |
| Purple (small) | Service point | The customer meter endpoint of a triplex service line |
| Blue (small) | Capacitor node | A node with a switched capacitor bank attached for reactive power / voltage support |
| Grey (small) | Standard node | All other distribution nodes — intermediate points along feeder lines |

**Degree** (shown in the Selected Element panel when a node is clicked) is the number of edges connected to that node. For example:

- A dead-end node at the tip of a lateral has degree 1
- A mid-line node has degree 2
- A three-way junction has degree 3
- A substation bus connecting to multiple feeder heads may have degree 5 or higher

Degree is useful for quickly understanding whether a node is a simple pass-through point, a branch junction, or a major hub.

---

### 5. Selected Element

When you click a node or edge, this panel displays its properties.

**For a node:**

| Field | Description |
|---|---|
| Name | The node identifier as it appears in the GLM/DSS model files |
| Type | The node classification (see Node Legend above) |
| Feeder | Which of the three feeders (S1, S2, S3) or sub-transmission group this node belongs to |
| Degree | Number of edges connected to this node |

**For an edge:**

| Field | Description |
|---|---|
| ID | The edge/line/transformer name from the model file |
| Type | The edge category (distribution line, switch, transformer, etc.) |
| From | The name of the source node |
| To | The name of the target node |
| (N.O.) | If shown in red next to the type, this switch is normally open in the base case |

---

## Power System Background

### What is the IEEE 9500-Node Test System?

The IEEE 9500-Node Test System is a synthetic distribution power system model developed by Pacific Northwest National Laboratory (PNNL) under the GridAPPS-D program, funded by the U.S. Department of Energy. It is an extension of the widely used IEEE 8500-Node Test Feeder and is designed to represent a realistic, modern distribution grid including legacy infrastructure, smart grid technologies, and distributed energy resources (DERs).

The model represents a section of a utility distribution system in the Pacific Northwest and contains approximately 9,500 voltage nodes across three feeders, a sub-transmission network, and multiple substations.

### The three feeders

**S1 — Modern grid with some renewables**
Located in the upper-right of the map. Contains a hospital microgrid with a backup diesel generator, a shopping center microgrid with an LNG engine, three wind turbines, and approximately 10% residential rooftop PV penetration.

**S2 — Future smart grid**
Located in the lower-left. Contains the New Neighborhood area with 100% rooftop PV penetration, three 200kW microturbines, and two 500kWh battery storage units enabling islanded microgrid operation.

**S3 — Legacy system**
Located in the upper-left (Old Town area). Contains a 3MW district steam/CHP plant, a 500kW solar farm, and no customer-side generation resources.

### Sub-transmission network

The three substations are interconnected at 69kV. The topology is:

```
115kV Source Bus
       |
  115/69kV transformer
       |
  S1 69kV bus ── 69kV line ── S2 69kV bus ── 69kV line ── S3 69kV bus
       |                              |                           |
  69/12.47kV LTC              69/12.47kV LTC              69/12.47kV LTC
       |                              |                           |
  S1 12.47kV feeder           S2 12.47kV feeder           S3 12.47kV feeder
```

### Normally-open switches and feeder boundaries

In the base case, the three feeders are operated as separate radial networks. Nine normally-open (N.O.) tie switches define the boundaries between feeders. These are shown in **red** in the visualization. During a fault or planned outage, operators can close one or more of these switches to transfer load between feeders.

### What is not in model_base.glm?

The GridLAB-D file `model_base.glm` contains only the distribution line and load data. The substation transformers, poletop voltage regulator transformers, DER step-up/step-down transformers, and switching devices are defined in the simulation platform at runtime using `model_dict.json` and the GridAPPS-D framework. The OpenDSS file `model_base.dss` is the complete, standalone model and is the source used to build this visualization.

---

## Files Used to Build This Visualization

| File | Role |
|---|---|
| `model_base.dss` | Primary source: all lines, transformers, and switches with full connectivity |
| `model_busxy.dss` | Geographic coordinates (longitude, latitude) for every node |
| `model_dict.json` | Identifies which switches are normally open in the base case |

---

## Known Limitations

- The feeder-to-substation assignments (S1/S2/S3) are determined by a breadth-first search from each substation's 12.47kV bus using only normally-closed switches. A small number of nodes near feeder boundaries may be assigned to an adjacent feeder.
- The visualization does not currently display simulation results (voltages, currents, loading). A future version can overlay these if exported from GridLAB-D or GridAPPS-D as a CSV.
- Very large hop counts (8+) on high-degree junction nodes may cause a brief pause while the neighborhood is computed.

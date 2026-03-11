flowchart TB

%% ======================
%% TOP-DOWN APPROACH
%% ======================

A[Feeder Demand Measurement] --> B[Allocation Factor AF]
B --> C[Initial Transformer Demand Estimate]

D[Transformer kVA Ratings] --> B

%% ======================
%% BOTTOM-UP APPROACH
%% ======================

E[Quantized DER States] --> F[State Feature Extraction]

F --> F1[ON Count]
F --> F2[Duty Cycle]
F --> F3[Transition Rates]
F --> F4[Coincidence / Diversity]
F --> F5[Feasible Power Bounds]

%% ======================
%% HOMOGENEITY MODELING
%% ======================

F --> G[Homogeneity / Heterogeneity Modeling]

G --> G1[Beta Distribution Parameters]
G --> G2[Moments of State Probabilities]
G --> G3[Population Variance]

%% ======================
%% FUSION LAYER
%% ======================

C --> H[Fusion Layer]

F1 --> H
F2 --> H
F3 --> H
F4 --> H
F5 --> H

G1 --> H
G2 --> H
G3 --> H

%% ======================
%% OUTPUT
%% ======================

H --> I[Refined Transformer Loading Estimate]
H --> J[Uncertainty / Confidence Bounds]
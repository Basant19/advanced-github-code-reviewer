"""from app.graph.workflow import review_graph

# Generate PNG diagram
png_bytes = review_graph.get_graph().draw_mermaid_png()

# Save file
with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as workflow_graph.png")"""


from app.graph.workflow import review_graph
from IPython.display import Image

png = review_graph.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(png)

print("Graph saved as workflow_graph.png")
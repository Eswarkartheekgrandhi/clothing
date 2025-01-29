import colorgram
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

colors = colorgram.extract('shirt.jpg', 5)
dominant_colors = [color.rgb for color in colors]

# 1. Print RGB values:
print("RGB Values:")
for i, color in enumerate(dominant_colors):
    print(f"Color {i+1}: RGB({color.r}, {color.g}, {color.b})")

# 2. Matplotlib Color Swatches:
fig, ax = plt.subplots()
for i, color in enumerate(dominant_colors):
    rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='black', facecolor=(color.r/255, color.g/255, color.b/255))
    ax.add_patch(rect)

ax.set_xlim(0, len(dominant_colors))
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
plt.title("Dominant Colors (Matplotlib)")
plt.show()


# 3. PIL Color Palette Image:
palette_width = 200
palette_height = 50
num_colors = len(dominant_colors)

palette_image = Image.new('RGB', (palette_width * num_colors, palette_height))
draw = ImageDraw.Draw(palette_image)

for i, color in enumerate(dominant_colors):
    x0 = i * palette_width
    x1 = (i + 1) * palette_width
    draw.rectangle([(x0, 0), (x1, palette_height)], fill=(color.r, color.g, color.b))

palette_image.save('color_palette.png')
palette_image.show()  # Or palette_image.save('color_palette.png') to save it
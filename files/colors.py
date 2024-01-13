import numpy as np
import colorsys


def generate_contrasting_colors(num_colors):
    hue = 0.0
    hue_increment = 1.0 / num_colors
    saturation = 0.7
    brightness = 0.7

    colors = []

    for _ in range(num_colors):
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        hue += hue_increment
        rgb_int = tuple(int(value * 255) for value in rgb)
        colors.append(rgb_int)

    return colors


num_colors = 80
color_array = generate_contrasting_colors(num_colors)
np.random.shuffle(color_array)

for c in color_array:
    print(f"{c},")

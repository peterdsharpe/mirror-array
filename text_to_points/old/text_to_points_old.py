import numpy as np
from PIL import Image, ImageFont, ImageDraw

### Inputs
text = "sei la mia luce".upper()
N = 294  # Number of points to use
resolution = 300

fontfile = "./fonts/AlegreyaSans-Thin.ttf"

### Get the font
font = ImageFont.truetype(fontfile, resolution)


### Generate each letter
def image_from_string(s: str) -> Image:
    temp_image = Image.new(
        '1',
        size=(1, 1)
    )
    temp_draw = ImageDraw.Draw(temp_image)
    size = temp_draw.textsize(
        text=s,
        font=font
    )

    image = Image.new(
        '1',
        size=size
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        xy=(0, 0),
        text=s,
        font=font,
        fill=1,
    )
    return image


### Figure out how many points should be allocated to each letter, dividing proportionally to area.
letter_images = [
    image_from_string(letter)
    for letter in text
]

letter_areas = np.array([
    np.sum(image)
    for image in letter_images
])

total_area = np.sum(letter_areas)

letter_n_points_ideal = N * letter_areas / total_area
letter_n_points = np.floor(letter_n_points_ideal).astype(int)
for i in range(N - np.sum(letter_n_points)):
    letter_n_points[np.argmax(letter_n_points_ideal - letter_n_points)] += 1

### Pick points for each letter
letter_points = []

x_origin = 0
y_origin = 0

for i in range(len(text)):
    image = letter_images[i]

    if letter_areas[i] == 0:
        letter_points.append(
            np.empty((0, 2))
        )
        x_origin += image.size[0]
        continue
    x, y = np.meshgrid(
        np.arange(image.size[0]),
        np.arange(image.size[1]),
    )
    valid_indices = np.arange(np.product(image.size))[np.ravel(image, order="F")]
    # point_indices = np.random.choice(
    #     a=valid_indices,
    #     size=letter_n_points[i]
    # )
    point_indices = valid_indices[np.linspace(0, len(valid_indices) - 1, letter_n_points[i]).astype(int)]

    x_points = np.ravel(x, order="F")[point_indices]
    y_points = np.ravel(y, order="F")[point_indices]

    letter_points.append(
        np.stack(
            [
                x_points + x_origin,
                y_points + y_origin,
            ],
            axis=1
        )
    )
    x_origin += image.size[0]

points = np.concatenate(letter_points, axis=0)
points = points / resolution
points[:, 1] *= -1
points[:, 1] += 0.5

image.save("test.png")
i = np.array(image)

### Display
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
plt.scatter(
    points[:, 0],
    points[:, 1],
    color="k",
    s=1
)
p.equal()
p.show_plot()

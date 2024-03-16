import matplotlib.pyplot as plt
import random
# Set up the pinhole camera parameters
list1 = [-4, -3, -2, -1, 0, 1, 2, 3, 4]



def pinHole (location):
    box_width = 10  # Width of the cardboard box
    hole_diameter = 0.1  # Diameter of the pinhole (tiny hole)
    # Create an empty canvas (our "box")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-box_width / 2, box_width / 2)
    ax.set_ylim(-box_width / 2, box_width / 2)
    ax.set_aspect('equal')
    # Draw the pinhole (a small circle)
    ax.add_patch(plt.Circle((location, location), hole_diameter / 2, color='black', fill=True))

    # Add the candle (a simple stick)
    ax.plot([location, location], [-box_width / 2, -box_width / 4], color='red', linewidth=2)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pinhole Camera Simulation')
for i in range(len(list1)):
    location = random.choice(list1)
    pinHole(location)



plt.show()

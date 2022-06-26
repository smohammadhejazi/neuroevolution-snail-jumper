import matplotlib.pyplot as plt


def plot_curve():
    with open('learning_curve.txt', "r") as file:
        lines = file.readlines()
        avg_values = []
        max_values = []
        min_values = []

        for line in lines:
            data = line.split(" ")
            avg_values.append(float(data[0]))
            max_values.append(float(data[1]))
            min_values.append(float(data[2]))

        x = list(range(1, len(lines) + 1))
        plt.plot(x, avg_values, 'bo-', label='Average')
        plt.plot(x, max_values, 'go-', label='Maximum')
        plt.plot(x, min_values, 'ro-', label='Minimum')

        plt.title("Fitness Graph")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plot_curve()

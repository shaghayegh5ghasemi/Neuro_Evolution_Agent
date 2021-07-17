import matplotlib.pyplot as plt

class DATA:
    def read_data(self, file_name):
        avg_data = []
        max_data = []
        min_data = []
        # Using readlines()
        file1 = open(file_name, 'r')
        Lines = file1.readlines()
        for line in Lines:
            data = line.split(" ")
            avg_data.append(float(data[0]))
            max_data.append(float(data[1]))
            min_data.append(float(data[2]))
        return avg_data, max_data, min_data


class PLOT():
    def plot(self, avg_data, max_data, min_data):
        avg_x = list(range(1, len(avg_data) + 1))
        max_x = list(range(1, len(max_data) + 1))
        min_x = list(range(1, len(min_data) + 1))
        plt.plot(avg_x, avg_data, label="Average Fitness")
        plt.plot(max_x, max_data, label="Maximum Fitness")
        plt.plot(min_x, min_data, label="Minimum Fitness")

        # naming the x axis
        plt.xlabel('Generation')
        # naming the y axis
        plt.ylabel('Fitness')
        # giving a title to my graph
        plt.title('Compare fitness info in different generations')
        # show a legend on the plot
        plt.legend()
        # function to show the plot
        plt.show()



if __name__ == '__main__':
    data = DATA()
    average, maximum, minimum = data.read_data('info.txt')
    plot = PLOT()
    plot.plot(average, maximum, minimum)
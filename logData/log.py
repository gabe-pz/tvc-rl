import csv
import matplotlib.pyplot as plt


def main() -> None:
    x_values: list[int] = []
    y_values: list[float] = []
    
    file = open("data/returnData.csv", "r", newline="")

    reader = csv.DictReader(file)

    for row in reader:
        x_values.append(int(row["episode"]))
        y_values.append(float(row["averageTotalReturn"]))

    file.close()
    
    plt.figure(figsize=(20,20))
    plt.plot(x_values, y_values)
    plt.xlabel("episode")
    plt.ylabel("averageTotalReturn")
    plt.show()


if(__name__ == "__main__"):
    main()
